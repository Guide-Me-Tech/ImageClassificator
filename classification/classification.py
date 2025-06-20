
# Download the dataset
import os
import os
import clip
import torch
import requests
from PIL import Image
from io import BytesIO
from config import logger, Config
import pandas as pd 
import os
from datetime import datetime
from models.output import Prediction
config = Config()



class ImageClassifier():
    def __init__(self):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        self.model, self.preprocess = clip.load(config.model_name, device)
        logger.info(f"Loaded CLIP model: {config.model_name}")
        self.classes = []
        self.classes_map = {}
        self.name_en_to_idx = {}
        

        
        
        
        
        self.load_classes("all.csv")
        logger.info(f"Loaded {len(self.classes)} classes")
        last_index = 0
        if not os.path.exists("usage.csv"):
            save_df = pd.DataFrame(columns=["time", "prediction", "image_path", "error"])
            save_df.to_csv("usage.csv", index=False)
            logger.info("Created new usage.csv file")
        else:
            last_index = len(pd.read_csv("usage.csv"))
            logger.info(f"Loaded existing usage.csv with {last_index} records")
        self.last_index = last_index
        self.device = device
        
        
    def load_classes(self, classes_file: str):
        logger.info(f"Loading classes from {classes_file}")
        df = pd.read_csv(classes_file)
        classes_map = {x["id"]: {"name_uz":  x["name_uz"], "name_ru": x["name_ru"], "name_en": x["name_en"]} for x in df.to_dict(orient="records")}
        # logger.info("Df type not ")
        self.name_en_to_idx = { x["name_en"]: x["id"] for x in df.to_dict(orient="records")}
        self.classes += df["name_en"].tolist()
        self.classes_map = classes_map
      
    def open_image(self, image_path_or_url):
        if image_path_or_url.startswith("http"):
            logger.debug(f"Downloading image from URL: {image_path_or_url}")
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
            # save to file 
            with open("image.png", "wb") as f:
                f.write(response.content)
            logger.debug("Saved downloaded image to image.png")
        else:
            logger.debug(f"Opening local image: {image_path_or_url}")
            image = Image.open(image_path_or_url)
        return image
    
    def prepare_inputs(self, image, classes):
        logger.debug("Preparing inputs for model")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(self.device)
        return image_input, text_inputs


    # Calculate features
    def calculate_features(self, image_input, text_inputs):
        logger.debug("Calculating image and text features")
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
        return image_features, text_features

    def calculate_similarity(self, image_features, text_features):
        logger.debug("Calculating similarity scores")
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity
    
    def predict(self, image_path_or_url, top_k=5):
        logger.info(f"Making prediction for {image_path_or_url}")
        image = self.open_image(image_path_or_url)
        image_input, text_inputs = self.prepare_inputs(image, self.classes)
        image_features, text_features = self.calculate_features(image_input, text_inputs)
        similarity = self.calculate_similarity(image_features, text_features)
        values, indices = similarity[0].topk(top_k)
        logger.info(f"Prediction complete, found top {top_k} matches")
        return values, indices

    
    def save_usage(self, prediction: Prediction, error: dict, image_path: str):
        
        logger.debug(f"Saving usage data for {image_path}")
        # append to usage.csv
        df = pd.DataFrame({"time": datetime.now(), "prediction": prediction.model_dump_json(), "image_path": image_path, "error": error}, index=[self.last_index])
        df.to_csv("usage.csv", mode="a", header=False, index=False)
        self.last_index += 1
        logger.debug("Usage data saved successfully")