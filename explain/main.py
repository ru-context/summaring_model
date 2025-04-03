from fastapi import FastAPI, UploadFile, File, HTTPException
from models import VectorDBModel, QASummarizer
from schemas import FileUpload, AnswerResponse, QuestionRequest
from fastapi.responses import JSONResponse

from database_manager import DatabaseManager
from functions import process_large_file, create_vector_db
from functions import search_in_db, extract_text_from_pdf, chunk_text

import uuid
import os

from typing import List

app = FastAPI()
db_manager = DatabaseManager()
vector_model = VectorDBModel()
qa_model = QASummarizer()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/", response_model=FileUpload)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Проверяем размер файла
        if os.path.getsize(file_path) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Обрабатываем файл
        try:
            chunks = process_large_file(file_path)
            if not chunks:
                raise HTTPException(status_code=400, detail="No text extracted from file")

            create_vector_db(chunks, file_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"file_id": file_id, "filename": file.filename}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Загрузка данных
        index = db_manager.load_index(request.file_id)
        texts = db_manager.load_texts(request.file_id)

        # Поиск релевантных фрагментов
        question_embedding = vector_model.encoder.encode([request.question])
        distances, indices = index.search(question_embedding, k=3) # type: ignore

        # Генерация ответа
        context = " ".join([texts[i] for i in indices[0]])
        qa_result = qa_model.generate_answer(request.question, context)

        return AnswerResponse(
            answer=qa_result['answer'],
            relevant_chunks=[texts[i] for i in indices[0]],
            confidence=qa_result['score']
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
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
