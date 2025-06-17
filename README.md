# Image Classification API

A FastAPI-based image classification service powered by OpenAI's CLIP (Contrastive Language-Image Pre-training) model. This service can classify images into over 200 predefined categories with confidence scores.

## What is this for?

This package provides an AI-powered image classification service that can:

- **Classify product images** into 200+ categories (electronics, clothing, home appliances, sports equipment, etc.)
- **Return confidence scores** for each classification
- **Support both local files and URLs** as image inputs
- **Track usage statistics** and maintain logs of all predictions
- **Provide RESTful API endpoints** for easy integration
- **Handle batch classification** with customizable top-k results

The service is designed for e-commerce platforms, inventory management systems, or any application that needs automated image categorization.

## Features

- 🤖 **CLIP-based classification** - Uses OpenAI's state-of-the-art vision-language model
- 🚀 **FastAPI framework** - High-performance async API with automatic documentation
- 📊 **Confidence scoring** - Get probability scores for each classification
- 📝 **Usage tracking** - Automatic logging of all predictions and errors
- 🐳 **Docker support** - Easy deployment with containerization
- 🔧 **Configurable categories** - Upload custom class lists via API
- 🖥️ **GPU acceleration** - Automatic CUDA detection and usage

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, but recommended for better performance)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Guide-Me-Tech/ImageClassificator
   cd ImageProcessing/classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The requirements.txt includes:
   - FastAPI for the web framework
   - PyTorch and torchvision for deep learning
   - OpenAI CLIP for image-text understanding
   - Pandas for data management
   - Pydantic for data validation

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   MODEL_NAME=ViT-B/32
   ```

4. **Prepare the categories:**
   The service comes with a pre-configured `category_list.txt` file containing 200+ product categories. You can modify this file or upload new categories via the API.

### Docker Installation

1. **Build the Docker image:**
   ```bash
   docker build -t image-classifier .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 image-classifier
   ```

   For GPU support:
   ```bash
   docker run --gpus all -p 8000:8000 image-classifier
   ```

## Usage

### Starting the Service

**Local development:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

### API Endpoints

#### 1. Classify Image
**POST** `/predict`

Upload an image file for classification.

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@your_image.jpg"
```

**Response:**
```json
{
  "prediction": {
    "classes_en": [
      {
        "class_name": "Smartphones",
        "confidence": 0.85
      },
      {
        "class_name": "Electronics",
        "confidence": 0.12
      }
    ],
    "classes_uz": []
  },
  "error": {}
}
```

#### 2. Upload Custom Categories
**POST** `/upload_classes`

Update the classification categories.

```bash
curl -X POST "http://localhost:8000/upload_classes" \
     -H "Content-Type: application/json" \
     -d '["Category 1", "Category 2", "Category 3"]'
```

#### 3. Get Current Categories
**GET** `/classes`

Retrieve the list of current classification categories.

```bash
curl -X GET "http://localhost:8000/classes"
```


### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff)

### Performance Considerations

- **GPU vs CPU**: The service automatically detects and uses CUDA if available, providing significant speed improvements
- **Image Size**: Larger images will take longer to process; consider resizing for faster inference
- **Batch Processing**: For multiple images, make parallel API calls rather than sequential ones

## Configuration

### Environment Variables

- `MODEL_NAME`: CLIP model variant to use (default: "ViT-B/32")
  - Options: "RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"

### Logging

The service includes comprehensive logging:
- Application logs are written to `app.log`
- Usage statistics are tracked in `usage.csv`
- Log levels can be configured in `config/logger_conf.py`

## File Structure

```
.
├── main.py                    # FastAPI application entry point
├── classification/            # Core classification module
│   ├── __init__.py
│   └── classification.py      # ImageClassifier implementation
├── models/                    # Pydantic data models
│   ├── __init__.py
│   └── output.py             # Response models
├── config/                    # Configuration files
│   ├── __init__.py
│   ├── config.py             # Application config
│   └── logger_conf.py        # Logging configuration
├── utils/                     # Utility functions
├── images/                    # Uploaded images storage
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
├── category_list.txt         # Default classification categories
└── usage.csv                # Usage tracking data
```

## Monitoring and Analytics

The service automatically tracks:
- **Prediction requests** with timestamps
- **Image file paths** and processing results  
- **Error logs** for failed predictions
- **Processing times** for performance monitoring

Access usage data through the generated `usage.csv` file or application logs.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU mode
2. **Model download fails**: Check internet connection and firewall settings
3. **Import errors**: Ensure all dependencies are installed correctly

### Debug Mode

Enable detailed logging by setting the log level to DEBUG in `config/logger_conf.py`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here] 