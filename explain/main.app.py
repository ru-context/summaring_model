from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import uuid
import os

app = FastAPI()

# Инициализация модели
try:
    print("Загрузка модели...")
    embedding_model = SentenceTransformer(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'
    ) # type: ignore
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {str(e)}")
    raise

# Конфигурация хранилища
FAISS_STORAGE = "faiss_storage"
os.makedirs(FAISS_STORAGE, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

def extract_text_from_pdf(file_path: str) -> str:
    """Извлекает текст из PDF файла"""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join(page.extract_text() or "" for page in reader.pages)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Сохраняем временный файл
        temp_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Извлекаем текст
        text = extract_text_from_pdf(temp_path)
        os.remove(temp_path)

        # Разбиваем на чанки
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size]]

        # Создаем эмбеддинги
        if not chunks:
            raise HTTPException(status_code=400, detail="Не удалось извлечь текст из PDF")

        print(f"Создание эмбеддингов для {len(chunks)} чанков...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)

        # Создаем индекс FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32)) # type: ignore

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
        if not all([
            os.path.exists(f"{FAISS_STORAGE}/{db_id}.index"),
            os.path.exists(f"{FAISS_STORAGE}/{db_id}_chunks.txt")
        ]):
            raise HTTPException(status_code=404, detail="База не найдена")

        # Загружаем индекс
        index = faiss.read_index(f"{FAISS_STORAGE}/{db_id}.index")

        # Загружаем чанки
        with open(f"{FAISS_STORAGE}/{db_id}_chunks.txt", "r", encoding="utf-8") as f:
            chunks = f.read().split("\n")

        # Кодируем вопрос
        question_embedding = embedding_model.encode([request.question])

        # Ищем в индексе
        k = min(3, len(chunks))
        distances, indices = index.search(question_embedding.astype(np.float32), k)

        return {
            "question": request.question,
            "results": [
                {
                    "text": chunks[i],
                    "score": float(dist),
                    "chunk_id": int(i)
                } for i, dist in zip(indices[0], distances[0])
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Сервер работает"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
