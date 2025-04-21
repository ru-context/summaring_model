import os
import uuid
import io
from pathlib import Path
from fastapi import UploadFile
import PyPDF2
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