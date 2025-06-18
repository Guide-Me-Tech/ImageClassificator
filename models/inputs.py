from pydantic import BaseModel
from typing import List
class NewClasess(BaseModel):
    classes_en: List[str]
    classes_ru: List[str]
    classes_uz: List[str]