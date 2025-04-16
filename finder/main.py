from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # Добавляем Pydantic для модели
import aiofiles
import PyPDF2
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
import os
import uuid

app = FastAPI()

# Инициализация моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
qa_pipeline = pipeline("question-answering", model="bert-base-uncased", tokenizer="bert-base-uncased", device=0 if torch.cuda.is_available() else -1)

# Инициализация FAISS индекса
dimension = 384  # Размерность эмбеддингов от all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
chunks = []  # Список для хранения текстовых чанков
file_id_to_chunks = {}  # Маппинг file_id к индексам чанков

# Модель для тела запроса
class AskRequest(BaseModel):
    file_id: str
    question: str

async def extract_text_from_pdf(file_path: str) -> str:
    """Извлечение текста из PDF файла."""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении PDF: {str(e)}")

async def extract_text_from_txt(file_path: str) -> str:
    """Извлечение текста из текстового файла."""
    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
        return await file.read()

def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """Разбиение текста на чанки."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def add_to_index(text_chunks: List[str], file_id: str):
    """Добавление чанков в FAISS индекс."""
    global index, chunks, file_id_to_chunks
    embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
    index.add(embeddings)
    start_idx = len(chunks)  # Используем глобальную переменную chunks для подсчёта
    file_id_to_chunks[file_id] = list(range(start_idx, start_idx + len(text_chunks)))
    chunks.extend(text_chunks)  # Добавляем новые чанки в глобальную переменную

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Эндпоинт для загрузки файла."""
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1].lower()
    file_path = f"uploads/{file_id}.{file_extension}"

    # Сохранение файла
    os.makedirs("uploads", exist_ok=True)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Извлечение текста
    if file_extension == "pdf":
        text = await extract_text_from_pdf(file_path)
    elif file_extension == "txt":
        text = await extract_text_from_txt(file_path)
    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла. Используйте PDF или TXT.")

    # Разбиение на чанки и добавление в индекс
    text_chunks = chunk_text(text)
    add_to_index(text_chunks, file_id)

    return JSONResponse(content={"file_id": file_id, "message": "Файл успешно обработан"}, status_code=200)

@app.post("/ask")
async def ask_question(request: AskRequest):
    """Эндпоинт для ответа на вопрос по загруженному файлу."""
    file_id = request.file_id
    question = request.question

    if file_id not in file_id_to_chunks:
        raise HTTPException(status_code=404, detail="Файл не найден")

    # Получение релевантных чанков
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(question_embedding, k=3)  # Поиск 3 ближайших чанков
    context = " ".join([chunks[idx] for idx in indices[0] if idx < len(chunks)])

    # Получение ответа
    result = qa_pipeline(question=question, context=context)
    return JSONResponse(content={"answer": result["answer"], "score": result["score"]}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)