from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from pydantic import BaseModel
from typing import Union
from models import QASystem
from langdetect import detect
from models import russian_model, english_model
from functions import extract_text_from_file, extract_text_from_url
from functions import split_text_into_chunks, extract_text_from_pdf
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Dict, Optional

import numpy as np
import requests
import logging
import torch
import uuid
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF QA System",
    description="System for processing PDFs and answering questions about their content",
    version="1.0"
)
qa_system = QASystem()

class TextInput(BaseModel):
    text: str

class UrlInput(BaseModel):
    url: str

class UploadResponse(BaseModel):
    book_id: str
    status: str
    chunks: int

class QuestionRequest(BaseModel):
    question: str
    book_id: str
    top_k: Optional[int] = 3

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    source_scores: List[float]

class SummaryRequest(BaseModel):
    book_id: str
    max_length: Optional[int] = 150

def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

@app.post("/upload/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Generate unique book ID
        book_id = str(uuid.uuid4())

        # Read and process file
        contents = await file.read()
        chunks_count = qa_system.process_pdf(book_id, contents)

        return {
            "book_id": book_id,
            "status": "success",
            "chunks": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = qa_system.answer_question(
            question=request.question,
            book_id=request.book_id,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/summarize/")
async def summarize_book(request: SummaryRequest):
    try:
        summary = qa_system.summarize_book(
            book_id=request.book_id,
            max_length=request.max_length
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/books/")
async def list_books():
    # This would need a method in VectorBookDatabase to list available books
    books = []
    for file in os.listdir("vector_storage"):
        if file.endswith(".faiss"):
            books.append(file.replace(".faiss", ""))
    return {"books": books}

@app.delete("/books/{book_id}")
async def delete_book(book_id: str):
    try:
        qa_system.vector_db.delete_book(book_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/process/")
async def process(
    request: Request,
    file: Union[UploadFile, None] = File(default=None),
    url: Union[str, None] = None,
):
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        try:
            json_data = await request.json()
            if "text" in json_data:
                text_input = TextInput(**json_data)
                content = text_input.text
                logger.info(f"Поступил запрос на суммаризацию текста: {content[:100]}...")
            elif "url" in json_data:
                url_input = UrlInput(**json_data)
                if not is_valid_url(url_input.url):
                    raise HTTPException(status_code=400, detail="Неверный URL")
                content = extract_text_from_url(url_input.url)
                logger.info(f"Поступил запрос на суммаризацию текста с URL: {url_input.url}")
            else:
                raise HTTPException(status_code=400, detail="Неверный формат JSON")
        except Exception as e:
            logger.error(f"Ошибка при обработке JSON: {e}")
            raise HTTPException(status_code=400, detail="Неверный формат JSON")


    elif content_type and "multipart/form-data" in content_type:
        if file:
            try:
                contents = await file.read()
                file_extension = os.path.splitext(file.filename)[1]  # type: ignore
                content = extract_text_from_file(contents, file_extension)
                logger.info(f"Текст из файла: {content[:100]}...")
            except Exception as e:
                logger.error(f"Ошибка при чтении файла: {e}")
                raise HTTPException(status_code=400, detail="Не удалось прочитать файл")
        else:
            raise HTTPException(status_code=400, detail="Файл не передан")


    elif url:
        try:
            if not is_valid_url(url):
                raise HTTPException(status_code=400, detail="Неверный URL")
            content = extract_text_from_url(url)
            logger.info(f"Поступил запрос на суммаризацию текста с URL: {url}")
        except Exception as e:
            logger.error(f"Ошибка при обработке URL: {e}")
            raise HTTPException(status_code=400, detail="Не удалось обработать URL")


    else:
        raise HTTPException(status_code=400, detail="Неверный Content-Type")
    try:
        lang = detect(content)
        logger.info(f"Определен язык: {lang}")
    except Exception as e:
        logger.error(f"Ошибка при определении языка: {e}")
        raise HTTPException(status_code=400, detail="Не удалось определить язык текста")

    russian_codes, english_codes = ['ru'], ['en']
    if lang in russian_codes:
        result = russian_model(content)
    elif lang in english_codes:
        result = english_model(content)
    else:
        logger.warning(f"Язык не поддерживается: {lang}")
        raise HTTPException(status_code=400, detail="Язык не поддерживается")

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
