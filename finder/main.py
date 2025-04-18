from fastapi import FastAPI, UploadFile, File, Form
from models.summarizer import summarize_text
from models.qa import get_answer
from utils.file_parser import extract_text
from utils.vector_store import DocumentStore
import uuid

app = FastAPI()
store = DocumentStore()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    text = await extract_text(file)
    doc_id = str(uuid.uuid4())
    summary = summarize_text(text)
    store.add_document(doc_id, text)
    return {"doc_id": doc_id, "summary": summary}

@app.post("/ask")
async def ask(doc_id: str = Form(...), question: str = Form(...)):
    if not store.has(doc_id):
        return {"error": "Document not found"}
    
    context = store.get_relevant_context(doc_id, question)
    answer = get_answer(context, question)
    return {"answer": answer}
