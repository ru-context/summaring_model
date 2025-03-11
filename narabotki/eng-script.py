import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, text, max_length=130, min_length=30):
        inputs = self.tokenizer(text, max_length=1024, truncation=True, padding="max_length", return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            repetition_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def split_long_text(self, text, max_tokens=1024):
        tokens = self.tokenizer.tokenize(text)
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    def summarize_long_text(self, text, max_length=130, min_length=30):
        chunks = self.split_long_text(text)
        summaries = [self.summarize(chunk, max_length=max_length, min_length=min_length) for chunk in chunks]
        return " ".join(summaries)


if __name__ == "__main__":
    summarizer = TextSummarizer()
    example_text = """
    The Transformer is a deep learning model introduced in 2017 by Vaswani et al. It relies entirely on self-attention mechanisms to process sequential data.
    Since its introduction, the Transformer has become a cornerstone in natural language processing, leading to significant advancements in machine translation,
    text summarization, and other NLP tasks. The model's architecture allows it to handle long-range dependencies more effectively than previous models.
    """

    summary = summarizer.summarize(example_text)
    print()
    print("Summary:", summary)
