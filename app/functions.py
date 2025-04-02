import fitz
import faiss
import uuid
import json
import requests
import chromadb
import numpy as np

from bs4 import BeautifulSoup # type: ignore
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

from typing import List, Dict, Tuple
from pathlib import Path

'''
Создаем класс для векторизации текста из pdf файлов и его дальнейшего сохранения в bd
'''
class VectorBookDatabase:
    def __init__(self, storage_dir: str = "vector_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.loaded_books = {}

    def _get_index_path(self, book_id: str) -> Path:
        return self.storage_dir / f"{book_id}.faiss"

    def _get_meta_path(self, book_id: str) -> Path:
        return self.storage_dir / f"{book_id}.json"

    def book_exists(self, book_id: str) -> bool:
        return self._get_index_path(book_id).exists()

    def add_book(self, book_id: str, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        if self.book_exists(book_id):
            raise ValueError(f"Book {book_id} already exists")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)

        # Generate embeddings
        embeddings = self.embedder.encode(chunks)
        dimension = embeddings.shape[1]

        # Create and save FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self._get_index_path(book_id)))

        # Save metadata
        metadata = {
            "chunks": chunks,
            "embeddings_shape": embeddings.shape,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        with open(self._get_meta_path(book_id), 'w') as f:
            json.dump(metadata, f)

        return len(chunks)

    def load_book(self, book_id: str) -> Tuple[faiss.Index, List[str]]:
        if book_id in self.loaded_books:
            return self.loaded_books[book_id]

        index = faiss.read_index(str(self._get_index_path(book_id)))
        with open(self._get_meta_path(book_id), 'r') as f:
            metadata = json.load(f)

        self.loaded_books[book_id] = (index, metadata["chunks"])
        return index, metadata["chunks"]

    def search_similar_chunks(self, query: str, book_id: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        index, chunks = self.load_book(book_id)
        query_embedding = self.embedder.encode([query])

        distances, indices = index.search(query_embedding, top_k)
        results = [chunks[i] for i in indices[0]]
        scores = [float(1 - distances[0][i]) for i in range(len(indices[0]))]  # Convert to similarity score

        return results, scores

    def delete_book(self, book_id: str) -> None:
        paths = [
            self._get_index_path(book_id),
            self._get_meta_path(book_id)
        ]
        for path in paths:
            if path.exists():
                path.unlink()
        self.loaded_books.pop(book_id, None)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF with proper error handling and formatting"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

'''
Тут код для корректной работы суммаризации и генерации адаптивной длины ответа
'''
def calculate_text_complexity(text: str) -> float:
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

def calculate_important_sentences(text: str) -> float:
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return 0

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        importance_scores = similarity_matrix.sum(axis=1)
        return importance_scores.mean()
    except:
        return 0

def classify_topic(text: str) -> str:
    technical_keywords = ["algorithm", "machine learning", "data science", "neural network"]
    if any(keyword in text.lower() for keyword in technical_keywords):
        return "technical"
    return "general"

def calculate_importance_with_bert(text: str) -> float:
    nlp = pipeline("feature-extraction", model="bert-base-uncased")
    try:
        features = nlp(text)
        importance = np.mean([np.mean(f) for f in features]) # type: ignore
        return importance # type: ignore
    except:
        return 0

def calculate_adaptive_summary_length(
    text: str,
    min_length: int = 50,
    max_length: int = 150,
    complexity_weight: float = 0.6,
    importance_weight: float = 0.2,
    topic_weight: float = 0.1,
    bert_weight: float = 0.1,
) -> int:
    complexity = calculate_text_complexity(text)
    importance = calculate_important_sentences(text)
    topic = classify_topic(text)
    bert_importance = calculate_importance_with_bert(text)

    complexity_norm = complexity
    importance_norm = importance
    topic_norm = 1.0 if topic == "technical" else 0.5
    bert_norm = bert_importance

    combined_score = (
        complexity_norm * complexity_weight +
        importance_norm * importance_weight +
        topic_norm * topic_weight +
        bert_norm * bert_weight
    )

    text_length = len(text.split())
    summary_length = int(text_length * 0.3 * combined_score)

    if summary_length < min_length:
        return min_length
    elif summary_length > max_length:
        return max_length
    else:
        return summary_length

def complete_sentence(text: str) -> str:
    if not text.endswith(('.', '!', '?')):
        last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punctuation != -1:
            text = text[:last_punctuation + 1]
    return text

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        raise ValueError(f"Ошибка при извлечении текста с URL: {e}")

def extract_text_from_file(file_bytes: bytes, file_extension: str) -> str:
    text = ""
    try:
        if file_extension == ".pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text() # type: ignore
        elif file_extension in [".txt", ".md"]:
            text = file_bytes.decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)
