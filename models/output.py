from pydantic import BaseModel

from typing import List, Dict




class ClassPrediction(BaseModel):
    class_name: str
    confidence: float


class Prediction(BaseModel):
    classes_en: List[ClassPrediction] = []
    classes_uz: List[ClassPrediction] = []
    classes_ru: List[ClassPrediction] = []

class Output(BaseModel):
    prediction: Prediction
    error: Dict[str, str]
    
