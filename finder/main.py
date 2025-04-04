import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI()
DATABASE_URL = "sqlite:///./files.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

model = SentenceTransformer('all-MiniLM-L6-v2') # type: ignore
class FileRecord(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)

class FileChunk(Base):
    __tablename__ = "file_chunks"
    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, index=True)
    chunk_number = Column(Integer)
    chunk_text = Column(Text)
    chunk_embedding = Column(String)

Base.metadata.create_all(bind=engine)
class SearchRequest(BaseModel):
    file_id: str
    query_text: str
    threshold: float = 0.5

class SearchResult(BaseModel):
    chunk_number: int
    chunk_text: str
    similarity_score: float

def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def serialize_embedding(embedding: np.ndarray) -> str:
    return ",".join(map(str, embedding.tolist()))

def deserialize_embedding(embedding_str: str) -> np.ndarray:
    return np.array(list(map(float, embedding_str.split(","))))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8")

        db = SessionLocal()
        file_id = str(uuid.uuid4())
        file_record = FileRecord(id=file_id, filename=file.filename)
        db.add(file_record)

        chunks = split_into_chunks(text)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk, convert_to_tensor=False)

            chunk_id = str(uuid.uuid4())
            chunk_record = FileChunk(
                id=chunk_id,
                file_id=file_id,
                chunk_number=i,
                chunk_text=chunk,
                chunk_embedding=serialize_embedding(embedding)
            )
            db.add(chunk_record)

        db.commit()
        db.close()

        return {"file_id": file_id, "filename": file.filename, "chunks_count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/", response_model=List[SearchResult])
async def search_text(request: SearchRequest):
    try:
        db = SessionLocal()
        file = db.query(FileRecord).filter(FileRecord.id == request.file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        chunks = db.query(FileChunk).filter(FileChunk.file_id == request.file_id).all()
        if not chunks:
            raise HTTPException(status_code=404, detail="No chunks found for this file")

        query_embedding = model.encode(request.query_text, convert_to_tensor=False)
        query_embedding = query_embedding.astype(np.float32)  # Приводим к float32

        results = []
        for chunk in chunks:
            chunk_embedding = deserialize_embedding(chunk.chunk_embedding)
            chunk_embedding = chunk_embedding.astype(np.float32)
            similarity = util.pytorch_cos_sim(query_embedding, chunk_embedding).item()

            if similarity >= request.threshold:
                results.append(SearchResult(
                    chunk_number=chunk.chunk_number,
                    chunk_text=chunk.chunk_text,
                    similarity_score=similarity
                ))

        results.sort(key=lambda x: x.similarity_score, reverse=True)

        db.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
