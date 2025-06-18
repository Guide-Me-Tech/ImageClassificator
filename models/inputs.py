from pydantic import BaseModel
from typing import List



class Class(BaseModel):
    id: int
    name_en: str
    name_ru: str
    name_uz: str


class NewClasess(BaseModel):
    classes: List[Class]