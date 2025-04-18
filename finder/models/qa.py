from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def get_answer(context: str, question: str) -> str:
    result = qa_pipeline(question=question, context=context)
    return result['answer']
