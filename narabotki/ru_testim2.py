import nltk
from transformers import pipeline

# Загружаем необходимые ресурсы NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Инициализируем суммаризатор
summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

# Пример текста
text = """
Различные слои используют разные функции активации, потому что у них разные цели (почти так же, как разные фреймворки используются для фронтенда и бэкенда).
Какую функцию активации использовать, зависит от одного — чего мы хотим добиться.
Эта функция может быть любой, на самом деле, но некоторые функции работают лучше.
Вот самые распространенные из них.
"""

# Функция для разбиения текста на части
def split_text(text, max_length=512):
    sentences = nltk.sent_tokenize(text, language='russian')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Разбиваем текст на части
chunks = split_text(text)

# Суммаризируем каждую часть
summaries = []
for chunk in chunks:
    summary = summarizer(chunk, max_length=60, min_length=30, do_sample=False)
    summaries.append(summary[0]['summary_text'])

# Объединяем результаты
final_summary = " ".join(summaries)
print(final_summary)
