import os
import uuid
import logging 
import torch 
import io
import PyPDF2

from enum import Enum
from pathlib import Path
from typing import List, Dict 
from fastapi import UploadFile
from docx import Document
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def generate_file_id() -> str:
    return str(uuid.uuid4())

def ensure_upload_dir(upload_dir: str = "uploads") -> None:
    os.makedirs(upload_dir, exist_ok=True)

def extract_text_from_file(file: UploadFile) -> str:
    file_extension = Path(file.filename).suffix.lower()
    
    try:
        if file_extension in [".txt", ".md", ".csv"]:
            contents = file.file.read()
            file.file.seek(0)
            return contents.decode("utf-8")

        elif file_extension == ".docx":
            content = file.file.read()
            file.file.seek(0)
            doc = Document(io.BytesIO(content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            return f"Text extraction not supported for {file_extension} files"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def save_extracted_text(text: str, file_id: str, upload_dir: str = "uploads") -> str:
    ensure_upload_dir(upload_dir)
    text_filename = f"{file_id}_extracted.txt"
    text_file_path = os.path.join(upload_dir, text_filename)
    
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(text)
    
    return text_file_path

def get_text_by_file_id(file_id: str, upload_dir: str = "uploads") -> str:
    text_filename = f"{file_id}_extracted.txt"
    text_file_path = os.path.join(upload_dir, text_filename)
    
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"File with ID {file_id} not found")
    
    with open(text_file_path, "r", encoding="utf-8") as file:
        return file.read()

def summarize_text(text: str, max_length: int = 150) -> str:
    cyrillic_count = sum(1 for char in text if 'а' <= char.lower() <= 'я')
    is_russian = cyrillic_count > len(text) * 0.3
    
    if is_russian:
        model_name = "IlyaGusev/rut5_base_sum_gazeta" 
    else:
        model_name = "sshleifer/distilbart-cnn-12-6" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    max_input_length = min(512, tokenizer.model_max_length) 
    
    if len(text) > max_input_length * 4:  
        beginning = text[:max_input_length]
        end = text[-max_input_length:]
        
        beginning_summary = summarizer(beginning, max_length=max_length//2, min_length=10, do_sample=False)
        end_summary = summarizer(end, max_length=max_length//2, min_length=10, do_sample=False)
        
        return beginning_summary[0]['summary_text'] + " " + end_summary[0]['summary_text']
    elif len(text) > max_input_length:
        text = text[:max_input_length]
    
    summary = summarizer(text, max_length=max_length, min_length=min(30, max_length//2), do_sample=False)
    return summary[0]['summary_text']


class SummarizationEnsemble:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.models = self._load_models()
        
    def _load_models(self) -> List[Dict]:
        """Загрузка только проверенных моделей для русского/английского"""
        models = [
            {
                'name': 'facebook/bart-large-cnn',
                'model': AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn'),
                'tokenizer': AutoTokenizer.from_pretrained('facebook/bart-large-cnn'),
                'lang': 'en',
                'weight': 0.6,
                'max_input': 1024
            },
            {
                'name': 'IlyaGusev/mbart_ru_sum_gazeta',
                'model': AutoModelForSeq2SeqLM.from_pretrained('IlyaGusev/mbart_ru_sum_gazeta'),
                'tokenizer': AutoTokenizer.from_pretrained('IlyaGusev/mbart_ru_sum_gazeta'),
                'lang': 'ru',
                'weight': 0.6,
                'max_input': 1024
            }
            # Убрана проблемная google/mt5-base
        ]
        
        for m in models:
            m['model'] = m['model'].to(self.device)
            m['model'].eval()
            
        return models
    
    def _prepare_text(self, text: str, model_name: str) -> str:
        """Предобработка текста для разных моделей"""
        if 'mbart' in model_name:
            return f"ru_RU {text}"  # Языковой префикс для mBART
        return text
    
    def summarize_single(self, text: str, model_info: Dict, params: dict) -> str:
        """Улучшенная генерация суммаризации"""
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        
        # Специальная обработка для русских текстов
        prepared_text = self._prepare_text(text, model_info['name'])
        
        inputs = tokenizer(
            prepared_text,
            max_length=model_info['max_input'],
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Параметры генерации для избежания артефактов
        generate_kwargs = {
            'max_length': params['max_length'],
            'min_length': params['min_length'],
            'num_beams': params['num_beams'],
            'length_penalty': params['length_penalty'],
            'early_stopping': True,
            'no_repeat_ngram_size': 3,  # Предотвращает повторения
            'bad_words_ids': [[tokenizer.eos_token_id]],  # Исключает спецтокены
        }
        
        # Особые настройки для mBART
        if 'mbart' in model_info['name']:
            generate_kwargs['forced_bos_token_id'] = tokenizer.lang_code_to_id['ru_RU']
        
        output = model.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def summarize(self, text: str, language: str = 'auto', **params) -> Dict:
        """Упрощенный и надежный ансамбль"""
        import time 
        start_time = time.time()
        lang = self.detect_language(text) if language == 'auto' else language
        
        results = []
        for model in self.models:
            if model['lang'] in [lang, 'multi']:
                try:
                    summary = self.summarize_single(text, model, params)
                    results.append({
                        'summary': summary,
                        'model': model['name'],
                        'weight': model['weight']
                    })
                except Exception as e:
                    self.logger.error(f"{model['name']} failed: {str(e)}")
        
        if not results:
            raise ValueError("No models could process this text")
        
        # Выбираем результат с максимальным весом
        best_result = max(results, key=lambda x: x['weight'])
        
        return {
            "summary": best_result['summary'],
            "model_used": best_result['model'],
            "language_detected": lang,
            "processing_time_ms": (time.time() - start_time) * 1000
        }