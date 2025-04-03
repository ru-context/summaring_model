import json
import faiss
from pathlib import Path
from typing import List

import os
import logging

class DatabaseManager:
    def __init__(self):
        self.base_dir = Path("database")
        self.base_dir.mkdir(exist_ok=True)

    def save_index(self, file_id: str, index: faiss.Index):
        faiss.write_index(index, str(self.base_dir / f"{file_id}.faiss"))

    def load_index(self, file_id: str) -> faiss.Index:
        return faiss.read_index(str(self.base_dir / f"{file_id}.faiss"))

    def save_texts(self, file_id: str, texts: List[str]) -> None:
        """Сохраняет текстовые чанки в JSON файл"""
        try:
            texts_path = self.base_dir / f"{file_id}_texts.json"
            with open(texts_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
            logging.info(f"Texts saved to {texts_path}")
        except Exception as e:
            logging.error(f"Error saving texts: {e}")
            raise

    def load_texts(self, file_id: str) -> List[str]:
        """Загружает текстовые чанки из JSON файла"""
        texts_path = self.base_dir / f"{file_id}_texts.json"
        if not texts_path.exists():
            raise FileNotFoundError(f"Texts file {texts_path} not found")

        try:
            with open(texts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {texts_path}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading texts: {e}")
            raise
