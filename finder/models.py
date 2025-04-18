from sqlalchemy import Column, String, Text, LargeBinary
from database import Base
import uuid
import pickle
import numpy as np
import faiss

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    text = Column(Text)
    chunks = Column(Text)
    embeddings = Column(LargeBinary)
    faiss_index = Column(LargeBinary)

    def set_embeddings(self, embeddings: np.ndarray):
        self.embeddings = pickle.dumps(embeddings)

    def get_embeddings(self) -> np.ndarray:
        return pickle.loads(self.embeddings) # type: ignore

    def set_faiss_index(self, index: faiss.Index):
        self.faiss_index = pickle.dumps(index)

    def get_faiss_index(self) -> faiss.Index:
        return pickle.loads(self.faiss_index) # type: ignore
