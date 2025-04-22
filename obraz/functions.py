import os
import uuid
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
        self.models = self._load_models()
        
    def _load_models(self) -> List[Dict]:
        """Загрузка предобученных моделей с более оптимальными весами"""
        models = [
            {
                'name': 'facebook/bart-large-cnn',
                'model': AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn'),
                'tokenizer': AutoTokenizer.from_pretrained('facebook/bart-large-cnn'),
                'lang': 'en',
                'weight': 0.5,
                'max_input': 1024
            },
            {
                'name': 'IlyaGusev/mbart_ru_sum_gazeta',
                'model': AutoModelForSeq2SeqLM.from_pretrained('IlyaGusev/mbart_ru_sum_gazeta'),
                'tokenizer': AutoTokenizer.from_pretrained('IlyaGusev/mbart_ru_sum_gazeta'),
                'lang': 'ru',
                'weight': 0.5,
                'max_input': 1024
            },
            {
                'name': 'google/mt5-base',
                'model': AutoModelForSeq2SeqLM.from_pretrained('google/mt5-base'),
                'tokenizer': AutoTokenizer.from_pretrained('google/mt5-base'),
                'lang': 'multi',
                'weight': 0.3,
                'max_input': 512
            }
        ]
        
        for m in models:
            m['model'] = m['model'].to(self.device)
            m['model'].eval()  # Переводим в режим inference
            
        return models

    def detect_language(self, text: str) -> str:
        """Улучшенное определение языка"""
        from langdetect import detect
        try:
            return detect(text)
        except:
            # Fallback на простой метод, если langdetect не сработал
            ru_chars = len([c for c in text.lower() if 'а' <= c <= 'я'])
            en_chars = len([c for c in text.lower() if 'a' <= c <= 'z'])
            return 'ru' if ru_chars > en_chars else 'en'
    
    def summarize_single(self, text: str, model_info: Dict, params: dict) -> str:
        """Генерация суммаризации одной моделью"""
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        
        inputs = tokenizer(
            text,
            max_length=model_info.get('max_input', 512),
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=params['max_length'],
            min_length=params['min_length'],
            length_penalty=params['length_penalty'],
            num_beams=params['num_beams'],
            early_stopping=True
        )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    def summarize(self, text: str, language: str = 'auto', **params) -> Dict:
        """
        Ансамблевая суммаризация
        
        Возвращает:
        {
            "summary": итоговая суммаризация,
            "model_used": имя лучшей модели,
            "language_detected": определенный язык,
            "all_summaries": все сгенерированные суммаризации
        }
        """
        import time
        start_time = time.time()
        
        # Определяем язык, если не указан явно
        lang = self.detect_language(text) if language == 'auto' else language
        lang = 'ru' if lang in ['ru', 'uk', 'be'] else 'en'  # Группируем славянские языки
        
        summaries = []
        model_names = []
        
        for model_info in self.models:
            if model_info['lang'] in [lang, 'multi']:
                try:
                    summary = self.summarize_single(text, model_info, params)
                    summaries.append(summary)
                    model_names.append(model_info['name'])
                except Exception as e:
                    print(f"Error with {model_info['name']}: {str(e)}")
        
        if not summaries:
            raise ValueError("No suitable models found for the detected language")
        
        # Выбираем лучшую summary по комбинации длины и веса модели
        best_idx = 0
        best_score = 0
        for i, (summary, model_name) in enumerate(zip(summaries, model_names)):
            model_weight = next(m['weight'] for m in self.models if m['name'] == model_name)
            # Оценка: вес модели * нормализованная длина summary
            score = model_weight * (len(summary) / params['max_length'])
            if score > best_score:
                best_score = score
                best_idx = i
        
        return {
            "summary": summaries[best_idx],
            "model_used": model_names[best_idx],
            "language_detected": lang,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "all_summaries": list(zip(model_names, summaries))  # Для отладки
        }
