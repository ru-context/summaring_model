from transformers import pipeline
from functions import calculate_adaptive_summary_length, complete_sentence
english_summarizer = pipeline("summarization", model="t5-small")
russian_summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

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
