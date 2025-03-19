from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
import fitz

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
    min_length: int = 30,
    max_length: int = 130,
    complexity_weight: float = 0.4,
    importance_weight: float = 0.3,
    topic_weight: float = 0.2,
    bert_weight: float = 0.1,
) -> int:
    """
    Универсальная функция для расчета адаптивной длины сокращения текста.

    :param text: Входной текст для суммаризации.
    :param min_length: Минимальная длина сокращения.
    :param max_length: Максимальная длина сокращения.
    :param complexity_weight: Вес сложности текста.
    :param importance_weight: Вес важности предложений.
    :param topic_weight: Вес тематики текста.
    :param bert_weight: Вес важности текста по BERT.
    :return: Адаптивная длина сокращения.
    """
    # Рассчитываем факторы
    complexity = calculate_text_complexity(text)
    importance = calculate_important_sentences(text)
    topic = classify_topic(text)
    bert_importance = calculate_importance_with_bert(text)

    # Нормализация факторов
    complexity_norm = complexity
    importance_norm = importance
    topic_norm = 1.0 if topic == "technical" else 0.5
    bert_norm = bert_importance

    # Комбинируем факторы с весами
    combined_score = (
        complexity_norm * complexity_weight +
        importance_norm * importance_weight +
        topic_norm * topic_weight +
        bert_norm * bert_weight
    )

    # Рассчитываем длину сокращения
    text_length = len(text.split())
    summary_length = int(text_length * 0.3 * combined_score)

    # Ограничиваем длину в пределах min_length и max_length
    if summary_length < min_length:
        return min_length
    elif summary_length > max_length:
        return max_length
    else:
        return summary_length

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() # type: ignore
    except Exception as e:
        raise ValueError(f"Ошибка при чтении PDF: {e}")
    return text
