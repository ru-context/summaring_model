import os
import torch 
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from functions import (
    generate_file_id,
    extract_text_from_file,
    save_extracted_text,
    get_text_by_file_id,
    summarize_text
)

from functions import (
    SummarizationEnsemble
)

ensemble = SummarizationEnsemble()

from schemas import UploadResponse, SummarizeResponse
from schemas import SummarizationResponse, SummarizationEnsemble, SummarizationRequest

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


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    try:
        params = {
            "max_length": request.max_length,
            "min_length": request.min_length,
            "num_beams": request.num_beams,
            "length_penalty": request.length_penalty
        }
        
        result = ensemble.summarize(
            text=request.text,
            language=request.language.value,
            **params
        )
        
        return {
            "summary": result["summary"],
            "model_used": result["model_used"],
            "language_detected": result["language_detected"],
            "processing_time_ms": result["processing_time_ms"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_loaded_models():
    """Возвращает список загруженных моделей и их параметры"""
    return [{
        "name": m["name"],
        "language": m["lang"],
        "weight": m["weight"],
        "max_input_length": m.get("max_input", 512)
    } for m in ensemble.models]

if __name__ == "__main__":
    import uvicorn
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)