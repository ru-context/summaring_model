from pydantic import BaseModel
from enum import Enum 
from typing import Optional, List, Dict 

import torch 

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    extracted_text: str
    success: bool
    message: Optional[str] = None

class SummarizeResponse(BaseModel):
    file_id: str
    summary: str
    success: bool
    message: Optional[str] = None


"""Схемы для суммаризации через ансамбль"""
class Language(str, Enum):
    AUTO = "auto"
    EN = "en"
    RU = "ru"

class SummarizationRequest(BaseModel):
    text: str
    language: Language = Language.AUTO
    max_length: int = 130
    min_length: int = 30
    num_beams: int = 4
    length_penalty: float = 2.0

class SummarizationResponse(BaseModel):
    summary: str
    model_used: str
    language_detected: str
    processing_time_ms: float

class SummarizationEnsemble:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = self._load_models()