
import dotenv
import os
dotenv.load_dotenv(override=True)








class Config:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "ViT-B/32")
        self.model_name_open_clip = os.environ.get("MODEL_NAME_OPEN_CLIP", "ViT-B-32-quickgelu")
        if os.getenv("USE_AUTH"):
            self.use_auth = True if os.getenv("USE_AUTH").lower() == "true" else False 
        else:
            self.use_auth = False
        self.auth_key = os.environ.get("AUTH_KEY", None)
        self.qdrant_host = os.environ.get("QDRANT_HOST", "qdrant")
        self.qdrant_port = os.environ.get("QDRANT_PORT", 6333)        
    
        
        

        