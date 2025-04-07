import uuid
import json
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from functions import extract_text_from_pdf, split_text_into_chunks, create_faiss_index
from schemas import QuestionRequest, SessionResponse, AnswerResponse
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from database import engine, Base, get_db
from models import Session
from crud import create_session, get_session, delete_session
from sqlalchemy.orm import Session as DBSession

app = FastAPI(
    title="PDF QA API",
    description="API для обработки PDF файлов и вопросно-ответной системы",
    version="1.0.0"
)

# Инициализация моделей
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
qa_model = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-base-squad2-distilled",
    tokenizer="deepset/xlm-roberta-base-squad2-distilled"
)

# Создание таблиц при старте
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.post("/upload/", response_model=SessionResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    try:
        text = extract_text_from_pdf(file.file)
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Не удалось извлечь текст из PDF"
            )

        chunks = split_text_into_chunks(text)
        embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
        index = create_faiss_index(chunks, embeddings)

        db_session = create_session(
            db=db,
            text=text,
            chunks=chunks,
            embeddings=embeddings,
            index=index
        )

        return {"session_id": str(db_session.id)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )

@app.post("/ask/{session_id}", response_model=AnswerResponse)
async def ask_question(
    session_id: str,
    request: QuestionRequest,
    db: DBSession = Depends(get_db)
):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный формат session_id"
        )

    db_session = get_session(db, session_uuid)
    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сессия не найдена"
        )

    try:
        question = request.question
        question_embedding = embedding_model.encode([question], convert_to_tensor=False)

        index = db_session.get_faiss_index()
        D, I = index.search(question_embedding, k=3)

        chunks = json.loads(db_session.chunks)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = " ".join(relevant_chunks)

        result = qa_model(question=question, context=context)

        return {"answer": result["answer"]}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обработке вопроса: {str(e)}"
        )

@app.delete("/session/{session_id}")
async def delete_session_endpoint(
    session_id: str,
    db: DBSession = Depends(get_db)
):
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный формат session_id"
        )

    if delete_session(db, session_uuid):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Сессия удалена"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сессия не найдена"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
