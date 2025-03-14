from transformers import pipeline
english_summarizer = pipeline("summarization", model="t5-small")
russian_summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

def russian_model(text: str, max_length: int = 130, min_length: int = 30):
    try:
        result = russian_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Ошибка при суммаризации текста: {e}"

def english_model(text: str, max_length: int = 130, min_length: int = 30):
    try:
        result = english_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Ошибка при суммаризации текста: {e}"
