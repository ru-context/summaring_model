from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str) -> str:
    if len(text) > 1024:
        text = text[:1024]
    result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']
