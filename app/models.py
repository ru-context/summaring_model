from transformers import pipeline
from vector_db import VectorBookDatabase
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from functions import calculate_adaptive_summary_length, complete_sentence
english_summarizer = pipeline("summarization", model="t5-small")
russian_summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

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
        # Или используем LangChain для более сложных сценариев
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                pipeline_kwargs={"max_length": 100}
            ),
            chain_type="stuff",
            retriever=self.vector_db.as_retriever()
        )

    def answer_question(self, question: str, book_id: str = None):
        # Получаем релевантные фрагменты
        chunks, _ = self.vector_db.search_similar_chunks(question, book_id)
        context = " ".join(chunks)

        # Получаем ответ на вопрос
        result = self.qa_pipeline(question=question, context=context)
        return {
            "answer": result['answer'],
            "score": result['score'],
            "context": chunks
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
