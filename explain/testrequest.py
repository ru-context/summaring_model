import requests

# Тест загрузки PDF
with open("test.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/upload_pdf/", files={"file": f})
    print(response.json())

# Тест вопроса
db_id = "ВАШ_ID_БАЗЫ"
response = requests.post(
    f"http://localhost:8000/question/{db_id}",
    json={"question": "Какой главный вопрос вселенной?"}
)
print(response.json())
