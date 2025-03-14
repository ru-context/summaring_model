import fitz

def calculate_summary_length(text: str, min_length: int = 30, max_length: int = 130) -> int:
    text_length = len(text.split())
    summary_length = int(text_length * 0.3)

    if summary_length < min_length:
        return min_length
    elif summary_length > max_length:
        return max_length
    else:
        return summary_length

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        # Открываем PDF из байтов
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
    except Exception as e:
        raise ValueError(f"Ошибка при чтении PDF: {e}")
    return text
