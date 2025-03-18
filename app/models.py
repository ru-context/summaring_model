from transformers import pipeline
from functions import calculate_adaptive_summary_length
english_summarizer = pipeline("summarization", model="t5-small")
russian_summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

def russian_model(text: str):
    try:
        summary_length = calculate_adaptive_summary_length(text)
        result = russian_summarizer(text, max_length=summary_length, min_length=int(summary_length * 0.7), do_sample=False)
        return result[0]['summary_text'] # type: ignore
    except Exception as e:
        return f"Ошибка при суммаризации текста: {e}"

def english_model(text: str):
    try:
        summary_length = calculate_adaptive_summary_length(text)
        result = english_summarizer(text, max_length=summary_length, min_length=int(summary_length * 0.7), do_sample=False)
        return result[0]['summary_text'] # type: ignore
    except Exception as e:
        return f"Ошибка при суммаризации текста: {e}"
