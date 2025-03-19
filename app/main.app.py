from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from pydantic import BaseModel
from typing import Union
from langdetect import detect
from models import russian_model, english_model
from functions import extract_text_from_file, extract_text_from_url
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextInput(BaseModel):
    text: str

class UrlInput(BaseModel):
    url: str

def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

@app.post("/process/")
async def process(
    request: Request,
    file: Union[UploadFile, None] = File(default=None),
    url: Union[str, None] = None,
):
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        try:
            json_data = await request.json()
            if "text" in json_data:
                text_input = TextInput(**json_data)
                content = text_input.text
                logger.info(f"Поступил запрос на суммаризацию текста: {content[:100]}...")
            elif "url" in json_data:
                url_input = UrlInput(**json_data)
                if not is_valid_url(url_input.url):
                    raise HTTPException(status_code=400, detail="Неверный URL")
                content = extract_text_from_url(url_input.url)
                logger.info(f"Поступил запрос на суммаризацию текста с URL: {url_input.url}")
            else:
                raise HTTPException(status_code=400, detail="Неверный формат JSON")
        except Exception as e:
            logger.error(f"Ошибка при обработке JSON: {e}")
            raise HTTPException(status_code=400, detail="Неверный формат JSON")


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


    elif url:
        try:
            if not is_valid_url(url):
                raise HTTPException(status_code=400, detail="Неверный URL")
            content = extract_text_from_url(url)
            logger.info(f"Поступил запрос на суммаризацию текста с URL: {url}")
        except Exception as e:
            logger.error(f"Ошибка при обработке URL: {e}")
            raise HTTPException(status_code=400, detail="Не удалось обработать URL")


    else:
        raise HTTPException(status_code=400, detail="Неверный Content-Type")
    try:
        lang = detect(content)
        logger.info(f"Определен язык: {lang}")
    except Exception as e:
        logger.error(f"Ошибка при определении языка: {e}")
        raise HTTPException(status_code=400, detail="Не удалось определить язык текста")

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
