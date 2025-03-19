from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from pydantic import BaseModel
from typing import Union
from langdetect import detect
from models import russian_model, english_model
from functions import extract_text_from_file
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/process/")
async def process(
    request: Request,
    file: Union[UploadFile, None] = File(default=None),
):
    content_type = request.headers.get("Content-Type")

    # Если Content-Type указывает на JSON
    if content_type == "application/json":
        try:
            json_data = await request.json()
            text_input = TextInput(**json_data)
            content = text_input.text
            logger.info(f"Поступил запрос на суммаризацию текста: {content[:100]}...")
        except Exception as e:
            logger.error(f"Ошибка при обработке JSON: {e}")
            raise HTTPException(status_code=400, detail="Неверный формат JSON")

    # Если Content-Type указывает на form-data (файл)
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

    # Если Content-Type не поддерживается
    else:
        raise HTTPException(status_code=400, detail="Неверный Content-Type")

    # Определение языка
    try:
        lang = detect(content)
        logger.info(f"Определен язык: {lang}")
    except Exception as e:
        logger.error(f"Ошибка при определении языка: {e}")
        raise HTTPException(status_code=400, detail="Не удалось определить язык текста")

    # Обработка текста в зависимости от языка
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
