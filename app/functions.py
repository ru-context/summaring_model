import fitz
import uuid
import requests
import numpy as np

from bs4 import BeautifulSoup # type: ignore
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from typing import List, Dict

'''
Создаем класс для векторизации текста из pdf файлов и его дальнейшего сохранения в bd
'''
class VectorBookDatabase:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="book_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add_book_chunks(self, book_id: str, chunks: List[str], metadata: List[Dict] = None):
        """Добавляет фрагменты книги в базу данных"""
        if not metadata:
            metadata = [{"book_id": book_id} for _ in chunks]

        embeddings = self.embedder.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )

    def search_similar_chunks(self, query: str, book_id: str = None, top_k: int = 5):
        """Ищет наиболее релевантные фрагменты текста"""
        query_embedding = self.embedder.encode(query).tolist()

        filters = {}
        if book_id:
            filters = {"book_id": book_id}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        return results['documents'], results['metadatas']

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
