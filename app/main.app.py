from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langdetect import detect
from models import russian_model, english_model
from functions import extract_text_from_pdf

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
class TextInput(BaseModel):
    text: str

@app.post("/process-text/")
async def process_text(text_input: TextInput):
    text = text_input.text
    try:
        lang = detect(text)
        logger.info(f"Поступил запрос на суммаризацию текста: {lang}")
    except Exception as e:
        logger.error(f"Ошибка при определении языка: {e}")
        raise HTTPException(status_code=400, detail="Не удалось определить язык текста")

    russian_codes, english_codes = ['ru'], ['en']
    if lang in russian_codes:
        result = russian_model(text)
    elif lang in english_codes:
        result = english_model(text)
    else:
        logger.warning(f"Язык не поддерживается: {lang}")
        raise HTTPException(status_code=400, detail="Язык не поддерживается")

    return {"result": result}


@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(contents)
        else:
            text = contents.decode("utf-8")

        logger.info(f"Текст из файла: {text[:100]}...")
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        raise HTTPException(status_code=400, detail="Не удалось прочитать файл")

    try:
        lang = detect(text)
        logger.info(f"Определен язык: {lang} для текста из файла")
    except Exception as e:
        logger.error(f"Ошибка при определении языка: {e}")
        raise HTTPException(status_code=400, detail="Не удалось определить язык текста")

    russian_codes, english_codes = ['ru'], ['en']
    if lang in russian_codes:
        result = russian_model(text)
    elif lang in english_codes:
        result = english_model(text)
    else:
        logger.warning(f"Язык не поддерживается: {lang}")
        raise HTTPException(status_code=400, detail="Язык не поддерживается")

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
