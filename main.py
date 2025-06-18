from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException, Query
from models.output import Output, ClassPrediction, Prediction
from fastapi.middleware.cors import CORSMiddleware
from models.inputs import NewClasess
from classification import ImageClassifier
from config import logger, Config
import time
from functools import wraps
from uuid import uuid4
from typing import List
import pandas as pd 
app = FastAPI(docs_url="/image/classification/docs")

config = Config()

classifier = ImageClassifier()

def timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function {func.__name__} took {duration:.2f} seconds to execute")
        return result
    return wrapper




async def check_key(auth_key: str = Header(...) ):
    if auth_key != config.auth_key:
        raise HTTPException(status_code=401, detail="Invalid authentication key")



@timer
async def predict_image(image: UploadFile = File(...), n_results: int = 5):
    try:
        logger.info(f"Processing image: {image.filename}")
        # save image to file
        filename = f"images/{uuid4()}_{image.filename}"
        with open(filename, "wb") as f:
            f.write(image.file.read())
        # predict
        values, indices = classifier.predict(filename, n_results)
        
        predictions = Prediction()
        for value, index in zip(values, indices):
            class_name = classifier.classes[index]
            class_index = classifier.name_en_to_idx[class_name]
            class_name_ru = classifier.classes_map[class_index]["name_ru"]
            class_name_uz = classifier.classes_map[class_index]["name_uz"]
            class_idx = classifier.name_en_to_idx[class_name]
            confidence = value.item()
            predictions.classes_en.append(ClassPrediction(class_name=class_name, confidence=confidence, idx=class_idx))
            predictions.classes_ru.append(ClassPrediction(class_name=class_name_ru, confidence=confidence, idx=class_idx))
            predictions.classes_uz.append(ClassPrediction(class_name=class_name_uz, confidence=confidence, idx=class_idx))
            logger.debug(f"Predicted class: {class_name} with confidence: {confidence:.2f}")

        logger.info(f"Successfully processed image: {filename}")
        
        # save to usage file
        
        classifier.save_usage(predictions, error={}, image_path=filename)
        return Output(prediction=predictions, error={})
    except Exception as e:
        logger.error(f"Error processing image {image.filename}: {str(e)}")
        classifier.save_usage(Prediction(), error={"error": str(e)}, image_path=filename)
        return Output(prediction=Prediction(), error={"error": str(e)})

if not config.use_auth:
    logger.info("No authentication")
    @app.post("/predict")
    async def predict(image: UploadFile = File(...), n_results: int = Query(default=5)):
        return await predict_image(image, n_results)
    

    @app.post("/upload_classes")
    def upload_classes(classes: NewClasess):
        classes = [ x.model_dump() for x in classes.classes]
        classes = pd.DataFrame(classes)
        all_classes = pd.read_csv("all.csv")
        classes_map = {x["id"]: {"name_uz":  x["name_uz"], "name_ru": x["name_ru"], "name_en": x["name_en"]} for x in all_classes.to_dict(orient="records")}

        for class_ in classes.classes:
            classes_map[class_["id"]] = {"name_uz": class_["name_uz"], "name_ru": class_["name_ru"], "name_en": class_["name_en"]}
                
        # convert to dataframe again
        classes_df = pd.DataFrame([[k, v["name_uz"], v["name_ru"], v["name_en"]] for k,v  in classes_map.items()])        
        classes_df.to_csv("all.csv", index=False)
        classifier.load_classes("all.csv")
        return {"message": "Classes uploaded successfully"}

    @app.get("/classes")
    def get_classes():
        return {"classes": classifier.classes_map}
else:
    logger.info("Authentication enabled")
    @app.post("/predict")
    async def predict(image: UploadFile = File(...), _ = Depends(check_key), n_results: int = Query(default=5)):
        return await predict_image(image, n_results)
    
    

    @app.post("/upload_classes")
    def upload_classes(classes: NewClasess):

        # classes = [ x.model_dump() for x in classes.classes]
        # new_classes = pd.DataFrame(classes)
        all_classes = pd.read_csv("all.csv")
        classes_map = {x["id"]: {"name_uz":  x["name_uz"], "name_ru": x["name_ru"], "name_en": x["name_en"]} for x in all_classes.to_dict(orient="records")}

        for class_ in classes.classes:
            classes_map[class_.id] = {"name_uz": class_.name_uz, "name_ru": class_.name_ru, "name_en": class_.name_en}
                
        # convert to dataframe again
        classes_df = pd.DataFrame([[k, v["name_uz"], v["name_ru"], v["name_en"]] for k,v  in classes_map.items()])    
        # set headers 
        classes_df.columns = ["id", "name_uz", "name_ru", "name_en"]
        classes_df.to_csv("all.csv", index=False)
        classifier.load_classes("all.csv")
        return {"message": "Classes uploaded successfully"}

    @app.get("/classes")
    def get_classes():
        return {"classes": classifier.classes_map}
