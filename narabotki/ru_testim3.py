import nltk
from transformers import pipeline

# Загружаем необходимые ресурсы NLTK
nltk.download('punkt')

# Инициализируем суммаризатор
summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

# Функция для разбиения текста на части
def split_text(text, max_length=512):
    sentences = nltk.sent_tokenize(text, language='russian')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Если текущее предложение слишком длинное, разбиваем его на части
        if len(sentence) > max_length:
            words = sentence.split()
            for i in range(0, len(words), max_length):
                chunk = " ".join(words[i:i + max_length])
                chunks.append(chunk)
        else:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Функция для суммаризации текста
def summarize_text(text, max_length=60, min_length=30, do_sample=False):
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Ошибка при суммаризации: {e}")
            summaries.append(chunk)  # Если не удалось суммаризировать, добавляем оригинальный текст

    final_summary = " ".join(summaries)
    return final_summary

# Функция для обработки большого файла
def summarize_large_file(input_file, output_file, max_length=60, min_length=30, do_sample=False):
    # Чтение файла
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Суммаризация текста
    final_summary = summarize_text(text, max_length=max_length, min_length=min_length, do_sample=do_sample)

    # Запись результата в файл
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(final_summary)

    print(f"Суммаризация завершена. Результат сохранен в файл: {output_file}")

# Пример использования
input_file = 'large_text.txt'  # Путь к большому текстовому файлу
output_file = 'summary.txt'   # Путь к файлу для сохранения суммаризации

summarize_large_file(input_file, output_file, max_length=60, min_length=30, do_sample=False)
