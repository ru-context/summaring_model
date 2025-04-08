from sqlalchemy.orm import Session
import json
from models import Session

def create_session(db: Session, text: str, chunks: list[str], embeddings, index):
    db_session = Session(
        text=text,
        chunks=json.dumps(chunks),
    )
    db_session.set_embeddings(embeddings)
    db_session.set_faiss_index(index)

    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_session(db: Session, session_id: str):
    return db.query(Session).filter(Session.id == session_id).first()

def delete_session(db: Session, session_id: str):
    db_session = get_session(db, session_id)
    if db_session:
        db.delete(db_session)
        db.commit()
        return True
    return False
