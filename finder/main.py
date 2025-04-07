import os
import uuid
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Инициализация FastAPI
app = FastAPI(
    title="PDF QA API",
    description="API для обработки PDF файлов и вопросно-ответной системы",
    version="1.0.0"
)

# Глобальные модели (загружаются один раз при старте)
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
qa_model = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-base-squad2-distilled",
    tokenizer="deepset/xlm-roberta-base-squad2-distilled"
)

# Хранилище сессий в памяти (в production замените на базу данных)
sessions: Dict[str, dict] = {}

# Модели запросов/ответов
class QuestionRequest(BaseModel):
    question: str

class SessionResponse(BaseModel):
    session_id: str

class AnswerResponse(BaseModel):
    answer: str

# Вспомогательные функции
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка при чтении PDF: {str(e)}"
        )

def create_faiss_index(text_chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# API endpoints
@app.post("/upload/", response_model=SessionResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Извлекаем текст из PDF
        text = extract_text_from_pdf(file.file)
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Не удалось извлечь текст из PDF"
            )

        # Разбиваем текст на чанки
        chunks = split_text_into_chunks(text)

        # Создаем эмбеддинги для чанков
        embeddings = embedding_model.encode(chunks, convert_to_tensor=False)

        # Создаем FAISS индекс для быстрого поиска
        index = create_faiss_index(chunks, embeddings)

        # Создаем новую сессию
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings,
            "index": index
        }

        return {"session_id": session_id}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )

@app.post("/ask/{session_id}", response_model=AnswerResponse)
async def ask_question(session_id: str, request: QuestionRequest):
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сессия не найдена"
        )

    try:
        session_data = sessions[session_id]
        question = request.question

        # Получаем эмбеддинг вопроса
        question_embedding = embedding_model.encode([question], convert_to_tensor=False)

        # Ищем наиболее релевантные чанки
        D, I = session_data["index"].search(question_embedding, k=3)  # top-3 чанка
        relevant_chunks = [session_data["chunks"][i] for i in I[0]]
        context = " ".join(relevant_chunks)

        # Получаем ответ от модели
        result = qa_model(question=question, context=context)

        return {"answer": result["answer"]}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке вопроса: {str(e)}"
        )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Сессия удалена"})
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сессия не найдена"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
