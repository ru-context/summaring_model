from transformers import pipeline
summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

text = """
Различные слои используют разные функции активации, потому что у них разные цели (почти так же, как разные фреймворки используются для фронтенда и бэкенда).
Какую функцию активации использовать, зависит от одного — чего мы хотим добиться.
Эта функция может быть любой, на самом деле, но некоторые функции работают лучше.
Вот самые распространенные из них.
"""

summary = summarizer(text, max_length=60, min_length=30, do_sample=False)
print(summary)
