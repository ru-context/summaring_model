from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
from models import russian_model, english_model

app = FastAPI()
class TextInput(BaseModel):
    text: str

@app.post("/process-text/")
async def process_text(text_input: TextInput):
    text = text_input.text
    try:
        lang = detect(text)
    except:
        raise HTTPException(status_code=400, detail="Не удалось определить язык текста")

    if lang == 'ru':
        result = russian_model(text)
    elif lang == 'en':
        result = english_model(text)
    else:
        raise HTTPException(status_code=400, detail="Язык не поддерживается")

    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
