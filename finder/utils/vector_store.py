from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.indexes = {}
        self.chunks = {}

    def chunk_text(self, text, max_length=256):
        words = text.split()
        return [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

    def add_document(self, doc_id, text):
        chunks = self.chunk_text(text)
        embeddings = self.model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        self.indexes[doc_id] = index
        self.chunks[doc_id] = chunks

    def has(self, doc_id):
        return doc_id in self.indexes

    def get_relevant_context(self, doc_id, query, k=3):
        query_vec = self.model.encode([query])
        D, I = self.indexes[doc_id].search(np.array(query_vec), k)
        relevant_chunks = [self.chunks[doc_id][i] for i in I[0]]
        return " ".join(relevant_chunks)
