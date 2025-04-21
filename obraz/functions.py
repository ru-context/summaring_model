import os
import uuid
import io
from pathlib import Path
from fastapi import UploadFile
import PyPDF2
from docx import Document

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