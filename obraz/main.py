import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from functions import (
    generate_file_id,
    extract_text_from_file,
    save_extracted_text,
    get_text_by_file_id,
    summarize_text
)
from schemas import UploadResponse, SummarizeResponse

app = FastAPI(title="File Upload and Text Extraction API")
UPLOAD_DIR = "uploads"


@app.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        file_id = generate_file_id()
        extracted_text = extract_text_from_file(file)
        text_file_path = save_extracted_text(extracted_text, file_id, UPLOAD_DIR)
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            extracted_text=extracted_text,
            success=True,
            message="File uploaded and processed successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "File Upload and Text Extraction API is running"}


@app.get("/summarize/{file_id}", response_model=SummarizeResponse)
async def summarize_file(file_id: str, max_length: int = 150):
    """
    Суммаризирует текст из файла по его file_id.
    Параметр max_length определяет максимальную длину суммаризации.
    """
    try:
        # Получаем текст из файла
        text = get_text_by_file_id(file_id, UPLOAD_DIR)
        
        # Суммаризируем текст
        summary = summarize_text(text, max_length)
        
        return SummarizeResponse(
            file_id=file_id,
            summary=summary,
            success=True,
            message="Text successfully summarized"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)