from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
from typing import List
from models import FileUpload, QuestionRequest, AnswerResponse
from functions import process_large_file, create_vector_db
from functions import search_in_db

app = FastAPI()
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
        # Поиск релевантных фрагментов
        relevant_chunks = search_in_db(request.question, request.file_id)

        # Здесь должна быть ваша логика суммаризации/ответа на вопрос
        # Для примера просто соединим релевантные фрагменты
        answer = " ".join(relevant_chunks)

        return {
            "answer": answer,
            "relevant_chunks": relevant_chunks
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
