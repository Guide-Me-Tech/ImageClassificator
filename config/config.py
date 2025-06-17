
import dotenv
import os
dotenv.load_dotenv(override=True)








class Config:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        

        