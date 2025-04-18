from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_sum_gazeta")
model = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/rut5_base_sum_gazeta")

def summarize_text(text: str) -> str:
    input_text = "summarize: " + text[:1500]
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = model.generate(**inputs, max_length=200, min_length=30)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
