from fastapi import FastAPI, UploadFile, File, HTTPException
from models import VectorDBModel, QASummarizer
from schemas import FileUpload, AnswerResponse, QuestionRequest
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
from pathlib import Path

from database_manager import DatabaseManager
from functions import process_large_file, create_vector_db
from functions import search_in_db, extract_text_from_pdf, chunk_text

import logging
import faiss
import uuid
import os

from typing import List

app = FastAPI()
db_manager = DatabaseManager()
vector_model = VectorDBModel()
qa_model = QASummarizer()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2') # type: ignore

@app.post("/upload/", response_model=FileUpload)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        # Сохраняем оригинальный файл
        file_path = upload_dir / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Обрабатываем файл
        chunks = process_large_file(str(file_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from file")

        # Сохраняем данные в векторную БД
        embeddings = model.encode(chunks)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Сохраняем ВСЕ данные
        db_manager = DatabaseManager()
        db_manager.save_index(file_id, index)
        db_manager.save_texts(file_id, chunks)  # Теперь это точно работает

        return {"file_id": file_id, "filename": file.filename}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        db_manager = DatabaseManager()
        texts = db_manager.load_texts(request.file_id)
        index = db_manager.load_index(request.file_id)

        # Увеличиваем k и используем косинусную близость
        question_embedding = model.encode([request.question])
        distances, indices = index.search(question_embedding, k=5)  # Больше чанков
        relevant_chunks = [texts[i] for i in indices[0]]

        # Логируем чанки для отладки
        logging.info(f"Relevant chunks: {relevant_chunks}")

        context = " ".join(relevant_chunks)
        qa_result = qa_model.generate_answer(request.question, context)

        # Фильтр по уверенности
        if qa_result["score"] < 0.1:
            return AnswerResponse(
                answer="Информация не найдена в документе.",
                relevant_chunks=relevant_chunks,
                confidence=qa_result["score"]
            )

        return AnswerResponse(
            answer=qa_result["answer"],
            relevant_chunks=relevant_chunks,
            confidence=qa_result["score"]
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/")
async def list_uploaded_files():
    """Просмотр загруженных файлов (для отладки)"""
    files = []
    for f in os.listdir(UPLOAD_DIR):
        files.append({
            "filename": f,
            "size": os.path.getsize(os.path.join(UPLOAD_DIR, f))
        })
    return {"files": files}

@app.get("/")
async def root():
    return {"message": "Text summarization and QA service"}
