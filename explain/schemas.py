from pydantic import BaseModel
from typing import Optional, List

class FileUpload(BaseModel):
    file_id: str
    filename: str

class QuestionRequest(BaseModel):
    file_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str
    relevant_chunks: Optional[List[str]] = None
    confidence: Optional[float] = None
