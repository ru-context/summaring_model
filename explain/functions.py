import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
from pathlib import Path

model = SentenceTransformer('all-MiniLM-L6-v2') # type: ignore

def extract_text_from_pdf(file_path: str) -> str:
    """Извлечение текста из PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(file_path: str) -> str:
    """Определение типа файла и извлечение текста"""
    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    # Можно добавить обработку других форматов (Word, Excel и т.д.)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_large_file(file_path: str, chunk_size: int = 1000) -> List[str]:
    """Улучшенная обработка файлов с автоматическим определением формата"""
    try:
        text = extract_text_from_file(file_path)
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    except Exception as e:
        raise ValueError(f"File processing failed: {str(e)}")

def create_vector_db(text_chunks: List[str], file_id: str) -> str:
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings) # type: ignore

    os.makedirs("database", exist_ok=True)
    index_path = f"database/{file_id}.faiss"
    faiss.write_index(index, index_path)

    return index_path

def search_in_db(question: str, file_id: str, k: int = 3) -> List[str]:
    """Поиск релевантных фрагментов в БД"""
    index_path = f"database/{file_id}.faiss"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector DB for file {file_id} not found")

    index = faiss.read_index(index_path)
    question_embedding = model.encode([question])

    distances, indices = index.search(question_embedding, k)
    return [f"Relevant chunk {i+1}" for i in indices[0]]
