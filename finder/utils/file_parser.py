import fitz  # PyMuPDF
from docx import Document

async def extract_text(file):
    if file.filename.endswith(".pdf"):
        return await extract_pdf(file)
    elif file.filename.endswith(".docx"):
        return await extract_docx(file)
    elif file.filename.endswith(".txt"):
        return (await file.read()).decode("utf-8")
    else:
        return "Unsupported file type."

async def extract_pdf(file):
    data = await file.read()
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

async def extract_docx(file):
    data = await file.read()
    with open("temp.docx", "wb") as f:
        f.write(data)
    doc = Document("temp.docx")
    return "\n".join(p.text for p in doc.paragraphs)

