from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException, Query
from models.output import Output, ClassPrediction, Prediction
from classification import ImageClassifier
from config import logger, Config
import time
from functools import wraps
from uuid import uuid4
from typing import List
import os
#app = FastAPI(docs_url="/image/classification/docs")
app = FastAPI(
    root_path="/image/classification/docs",
    docs_url="/",  # This will make docs available at /image/classification/docs/
    redoc_url=None,  # Or set to another if needed
    openapi_url="/openapi.json"
)
config = Config()

classifier = ImageClassifier()
if not os.path.exists("images"):
    os.makedirs("images")
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
            confidence = value.item()
            predictions.classes_en.append(ClassPrediction(class_name=class_name, confidence=confidence))
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
else:
    logger.info("Authentication enabled")
    @app.post("/image/classification/predict", response_class=Output)
    async def predict(image: UploadFile = File(...), _ = Depends(check_key), n_results: int = Query(default=5)):
        return await predict_image(image, n_results)


@app.post("/upload_classes")
def upload_classes(classes: List[str]):
    with open("categories_list_new.txt", "w") as f:
        for class_ in classes:
            f.write(class_ + "\n")
    
    classifier.load_classes("categories_list_new.txt")
    return {"message": "Classes uploaded successfully"}

@app.get("/classes")
def get_classes():
    return {"classes": classifier.classes}
