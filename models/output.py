from pydantic import BaseModel

from typing import List, Dict




class ClassPrediction(BaseModel):
    category_name: str
    confidence: float
    category_id: int
    idx: int

class ClassSimilarity(BaseModel):
    sim_score: float
    product_name: str
    product_id: int

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
    
