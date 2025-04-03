from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch

class VectorDBModel:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') # type: ignore
        self.dimension = 384  # Размерность для all-MiniLM-L6-v2

    def create_index(self, texts: List[str]) -> faiss.Index:
        embeddings = self.encoder.encode(texts)
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings) # type: ignore
        return index

class QASummarizer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device
        )
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device
        )

    def generate_answer(self, question: str, context: str) -> Dict:
        return self.qa_pipeline(question=question, context=context) # type: ignore

    def summarize(self, text: str, max_length: int = 150) -> str:
        result = self.summarizer(text, max_length=max_length)
        return result[0]['summary_text'] # type: ignore
