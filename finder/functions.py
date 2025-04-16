import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from fastapi import HTTPException

def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # type: ignore
            detail=f"Ошибка при чтении PDF: {str(e)}"
        )

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings должны быть numpy array")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings должны быть 2D массивом")
    if embeddings.shape[0] == 0:
        raise ValueError("Передан пустой массив embeddings")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    if not index.is_trained:
        index.train(embeddings)  
    index.add(embeddings)
    assert index.ntotal == embeddings.shape[0], "Не все эмбеддинги добавились в индекс"
    return index

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_faiss_index(self) -> faiss.Index:
    try:
        return pickle.loads(self.faiss_index)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки индекса: {str(e)}")