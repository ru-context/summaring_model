from transformers import pipeline
from functions import calculate_adaptive_summary_length, complete_sentence
english_summarizer = pipeline("summarization", model="t5-small", timeout=30)
russian_summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta", timeout=30)

'''
Класс для работы Explainable function
'''
class QASystem:
    def __init__(self):
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_chunks = []
        self.metadata = []

    def add_book_chunks(self, book_id: str, chunks: list[str]):
        self.text_chunks.extend(chunks)
        self.metadata.extend([{"book_id": book_id} for _ in chunks])

    def _get_embeddings(self, texts: list[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def answer_question(self, question: str, book_id: str = None):
        # Получаем эмбеддинги для всех чанков
        chunk_embeddings = self._get_embeddings(self.text_chunks)
        question_embedding = self._get_embeddings([question])

        # Вычисляем схожесть
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]

        # Фильтруем по book_id если нужно
        valid_indices = [
            i for i, meta in enumerate(self.metadata)
            if book_id is None or meta["book_id"] == book_id
        ]
        top_indices = np.argsort(similarities[valid_indices])[-5:][::-1]

        # Получаем контекст
        context = " ".join([self.text_chunks[valid_indices[i]] for i in top_indices])

        # Получаем ответ
        result = self.qa_pipeline(question=question, context=context)
        return {
            "answer": result['answer'],
            "score": result['score'],
            "context": [self.text_chunks[valid_indices[i]] for i in top_indices]
        }

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
