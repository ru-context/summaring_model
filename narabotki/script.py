from transformers import T5ForConditionalGeneration, T5Tokenizer
model_name = "sberbank-ai/ruT5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

text = "В России началась весна. Погода становится теплее, и люди начинают выходить на улицы."
input_text = "summarize: " + text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=50,
    min_length=10,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=2,
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("T5 Суммаризация:", summary)
