from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    extracted_text: str
    success: bool
    message: Optional[str] = None