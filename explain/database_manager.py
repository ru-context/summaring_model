import json
import faiss
from pathlib import Path
from typing import List

class DatabaseManager:
    def __init__(self):
        self.base_dir = Path("database")
        self.base_dir.mkdir(exist_ok=True)

    def save_index(self, file_id: str, index: faiss.Index):
        faiss.write_index(index, str(self.base_dir / f"{file_id}.faiss"))

    def load_index(self, file_id: str) -> faiss.Index:
        return faiss.read_index(str(self.base_dir / f"{file_id}.faiss"))

    def save_texts(self, file_id: str, texts: List[str]):
        with open(self.base_dir / f"{file_id}_texts.json", "w") as f:
            json.dump(texts, f)

    def load_texts(self, file_id: str) -> List[str]:
        with open(self.base_dir / f"{file_id}_texts.json", "r") as f:
            return json.load(f)
