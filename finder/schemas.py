from pydantic import BaseModel
class QuestionRequest(BaseModel):
    question: str

class SessionResponse(BaseModel):
    session_id: str

class AnswerResponse(BaseModel):
    answer: str
