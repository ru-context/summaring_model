import faiss
from PyPDF2 import PdfReader
from fastapi import HTTPException

def extract_text_from_pdf(file_stream):
    try:
        pdf_reader = PdfReader(file_stream)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # type: ignore
            detail=f"Ошибка при чтении PDF: {str(e)}"
        )

def create_faiss_index(text_chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings) # type: ignore
    return index

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks
