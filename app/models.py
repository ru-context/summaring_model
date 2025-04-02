from transformers import pipeline, AutoTokenizer, AutoModel
from functions import calculate_adaptive_summary_length, complete_sentence, VectorBookDatabase
english_summarizer = pipeline("summarization", model="t5-small", timeout=30)
russian_summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta", timeout=30)

from typing import List, Dict, Optional

'''
Класс для работы Explainable function
'''
class QASystem:
    def __init__(self):
        self.vector_db = VectorBookDatabase()
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    def process_pdf(self, book_id: str, pdf_bytes: bytes) -> int:
        """Process PDF and store in vector database"""
        text = extract_text_from_pdf(pdf_bytes)
        return self.vector_db.add_book(book_id, text)

    def answer_question(self, question: str, book_id: str, top_k: int = 3) -> Dict:
        """Answer question based on book content"""
        similar_chunks, scores = self.vector_db.search_similar_chunks(question, book_id, top_k)
        context = " ".join(similar_chunks)

        result = self.qa_pipeline(question=question, context=context)

        return {
            "answer": result['answer'],
            "confidence": float(result['score']),
            "sources": similar_chunks,
            "source_scores": scores
        }

    def summarize_book(self, book_id: str, max_length: int = 150) -> str:
        """Generate summary for the entire book"""
        _, chunks = self.vector_db.load_book(book_id)
        full_text = " ".join(chunks)

        # For very long texts, we summarize in parts
        if len(full_text.split()) > 1024:
            summaries = []
            for i in range(0, len(chunks), 10):
                batch = " ".join(chunks[i:i+10])
                summary = self.summarizer(batch, max_length=max_length, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            full_text = " ".join(summaries)

        result = self.summarizer(full_text, max_length=max_length, min_length=30, do_sample=False)
        return result[0]['summary_text']

'''
Функции для простой суммаризации текста
'''
def summarize_long_text(text: str, summarizer, max_tokens: int = 500) -> str:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_tokens:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    summaries = []
    for chunk in chunks:
        summary_length = calculate_adaptive_summary_length(chunk)
        result = summarizer(chunk, max_length=summary_length, min_length=int(summary_length * 0.5), do_sample=False)
        summaries.append(result[0]['summary_text'])

    return complete_sentence(" ".join(summaries)) # type: ignore

def russian_model(text: str):
    try:
        return summarize_long_text(text, russian_summarizer)
    except Exception as e:
        return f"Ошибка при суммаризации текста: {e}"

def english_model(text: str):
    try:
        return summarize_long_text(text, english_summarizer)
    except Exception as e:
        return f"Ошибка при суммаризации текста: {e}"
