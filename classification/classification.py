
# Download the dataset
import os
from qdrant_client.models import Filter, FieldCondition, MatchAny, SearchParams
from qdrant_client import QdrantClient
from PIL import Image
import torch
import open_clip
import requests
from io import BytesIO
from config import logger, Config
import pandas as pd 
from datetime import datetime
from models.output import Prediction, Similarity

config = Config()


class ImageClassifier():
    def __init__(self):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = QdrantClient(host="localhost", port=6333)
        logger.info(f"Using device: {device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model_name or "ViT-B-32-quickgelu", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(config.model_name or "ViT-B-32-quickgelu")
        self.model = self.model.to(device)
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
        text_inputs = self.tokenizer([f"a photo of a {c}" for c in classes]).to(self.device)
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

    def get_top_categories(self, image_path_or_url, top_k=3):
        values, indices = self.predict(image_path_or_url, top_k=top_k)
        top_names = [self.classes[i] for i in indices.tolist()]
        return top_names

    def get_query_vector(self, image_path_or_url):
        query_image = self.preprocess(Image.open(image_path_or_url).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_vector = self.model.encode_image(query_image)
            if query_vector.shape[0] == 1:
                query_vector = query_vector.squeeze(0)
            else:
                logger.error(f"Unexpected tensor shape: {query_vector.shape}")
                exit(1)
            norm = query_vector.norm()
            if norm > 0:
                query_vector = query_vector / norm
            else:
                logger.error("Error: Query vector has zero norm.")
                exit(1)
        return query_vector

    def qdrant_search(self, image_path_or_url, top_k_categs=3):
        top_categs = self.get_top_categories(image_path_or_url, top_k_categs)
        logger.info(f"Top categories from classifier: {top_categs}")
        query_vector = self.get_query_vector(image_path_or_url)
        
        category_filter = Filter(
            must=[
                FieldCondition(
                    key="categories",
                    match=MatchAny(any=top_categs)
                )
            ]
        )

        results = self.client.query_points(
            collection_name="smartbazar_products",
            query=query_vector.tolist(),
            query_filter=category_filter,
            search_params=SearchParams(hnsw_ef=128),
            limit=10
        )
        logger.info(f"Performed QDrant Similiarity Search from {top_categs} | Retrieved results")

        return [
            {
                "score": hit.score,
                "product_name": hit.payload.get("product_name")
            }
            for hit in results.points
        ]   

    def save_usage(self, prediction: Prediction, similars: Similarity, error: dict, image_path: str):
        
        logger.debug(f"Saving usage data for {image_path}")
        # append to usage.csv
        df = pd.DataFrame({"time": datetime.now(), "prediction": prediction.model_dump_json(), "similars": similars.model_dump_json(), "image_path": image_path, "error": error}, index=[self.last_index])
        df.to_csv("usage.csv", mode="a", header=False, index=False)
        self.last_index += 1
        logger.debug("Usage data saved successfully")