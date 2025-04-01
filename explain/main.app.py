import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import uuid

app = FastAPI()

# Инициализация модели (используем sentence-transformers напрямую)
try:
    # Явно указываем использование PyTorch
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

# Хранилище для индексов
faiss_indices = {}
FAISS_STORAGE = "faiss_storage"
os.makedirs(FAISS_STORAGE, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

def extract_text_from_pdf(file):
    """Извлечение текста из PDF"""
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Сохраняем временный файл
        temp_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        # Извлекаем текст
        text = extract_text_from_pdf(temp_path)
        os.remove(temp_path)  # Удаляем временный файл

        # Разбиваем текст на чанки
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        # Создаем эмбеддинги
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)

        # Создаем FAISS индекс
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

        # Сохраняем индекс
        db_id = str(uuid.uuid4())
        faiss.write_index(index, f"{FAISS_STORAGE}/{db_id}.index")

        # Сохраняем чанки
        with open(f"{FAISS_STORAGE}/{db_id}_chunks.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))

        return {"db_id": db_id, "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question/{db_id}")
async def ask_question(db_id: str, request: QuestionRequest):
    try:
        # Проверяем существование файлов
        if not os.path.exists(f"{FAISS_STORAGE}/{db_id}.index"):
            raise HTTPException(status_code=404, detail="База не найдена")

        # Загружаем индекс
        index = faiss.read_index(f"{FAISS_STORAGE}/{db_id}.index")

        # Загружаем чанки
        with open(f"{FAISS_STORAGE}/{db_id}_chunks.txt", "r", encoding="utf-8") as f:
            chunks = f.read().split("\n")

        # Кодируем вопрос
        question_embedding = embedding_model.encode([request.question])

        # Ищем в индексе
        k = min(3, len(chunks))  # Не больше чем есть чанков
        distances, indices = index.search(question_embedding.astype(np.float32), k)

        # Формируем ответ
        results = []
        for i, dist in zip(indices[0], distances[0]):
            results.append({
                "text": chunks[i],
                "score": float(dist),
                "chunk_id": int(i)
            })

        return {"question": request.question, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
