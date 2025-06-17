
import dotenv
import os
dotenv.load_dotenv(override=True)








class Config:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.use_auth = True if os.getenv("USE_AUTH").lower() == "true" else False
        self.auth_key = os.getenv("AUTH_KEY")
        print(os.getenv("MODEL_NAME"))
        print(os.getenv("USE_AUTH"))
        print(os.getenv("AUTH_KEY"))
        print(self.use_auth)
        
    
        
        

        