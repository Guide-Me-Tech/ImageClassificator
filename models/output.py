from pydantic import BaseModel

from typing import List, Dict




class ClassPrediction(BaseModel):
    class_name: str
    confidence: float

class ClassSimilarity(BaseModel):
    sim_score: float
    product_name: str

class Prediction(BaseModel):
    classes_en: List[ClassPrediction] = []
    classes_uz: List[ClassPrediction] = []
    classes_ru: List[ClassPrediction] = []

class Similarity(BaseModel):
    similar_products: List[ClassSimilarity] = []
    
class Output(BaseModel):
    prediction: Prediction
    similarity: Similarity
    error: Dict[str, str]
    
