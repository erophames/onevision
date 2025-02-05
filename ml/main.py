from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet_v2 import preprocess_input
import json
import io
import logging
import re
from PIL import Image, ImageFilter, ImageOps
import base64
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom metric classes required for loading the model
class SparsePrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

class SparseRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

class PathogenPredictor:
    IMG_SIZE = 300
    TTA_STEPS = 5  # Test-time augmentation steps

    BASE_COST = 1.0
    LETTERBOX_COST = 0.5
    COST_PER_TTA_STEP = 0.2

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_labels = None

    def load_model(self):
        """Load the trained model and class labels"""
        try:
            # Load model with custom metrics
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    'SparsePrecision': SparsePrecision,
                    'SparseRecall': SparseRecall
                }
            )

            # Load class labels from associated JSON file
            with open(f"{self.model_path}_class_names.json", 'r') as f:
                self.class_labels = json.load(f)

            logger.info("Model and labels loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_data):
        """Make prediction on image bytes"""
        try:
            # Check if model is loaded
            if self.model is None or self.class_labels is None:
                raise ValueError("Model not loaded - call load_model() first")

            processing_metadata = {
                'letterboxed': False,
                'tta_steps': self.TTA_STEPS
            }

            # Load image from bytes
            img = Image.open(io.BytesIO(image_data))
            img = ImageOps.exif_transpose(img)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Smart image scaling with aspect ratio preservation
            aspect_ratio = img.width / img.height
            if aspect_ratio > 2 or aspect_ratio < 0.5:
                logger.info("Image has extreme ratios - letterboxing")
                processing_metadata['letterboxed'] = True
                img.thumbnail((self.IMG_SIZE*2, self.IMG_SIZE*2))
                background = Image.new('RGB', (self.IMG_SIZE, self.IMG_SIZE), (0,0,0))
                offset = (
                    (self.IMG_SIZE - img.width) // 2,
                    (self.IMG_SIZE - img.height) // 2
                )
                background.paste(img, offset)
                img = background
            else:
                logger.info("Scaling image proportionally...")
                if max(img.size) > self.IMG_SIZE * 4:
                    img = img.resize(
                        (self.IMG_SIZE*4, self.IMG_SIZE*4),
                        resample=Image.Resampling.LANCZOS
                    )
                    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

                img = img.resize(
                    (self.IMG_SIZE, self.IMG_SIZE),
                    resample=Image.Resampling.BILINEAR
                )

            # Convert to array and preprocess
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)

            # Convert the processed image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Test-time augmentation
            predictions = []
            for _ in range(self.TTA_STEPS):
                augmented = self._apply_tta_augmentation(img_array)
                predictions.append(self.model.predict(augmented, verbose=0))

            # Process predictions
            avg_prediction = np.mean(predictions, axis=0)
            conf = np.max(avg_prediction)
            class_idx = np.argmax(avg_prediction)
            class_label = self.class_labels[class_idx]

            # Parse class label
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"
            plant, disease = re.match(pattern, class_label).groups()
            plant = re.sub(r"_", " ", plant)
            disease = re.sub(r"_", " ", disease).title()

            logger.info(plant)
            logger.info(disease)

            credits_used = self.BASE_COST
            if processing_metadata['letterboxed']:
                credits_used += self.LETTERBOX_COST
            credits_used += self.TTA_STEPS * self.COST_PER_TTA_STEP

            return {
                "plant": plant,
                "disease": disease,
                "confidence": float(conf),
                "processed_image": img_base64,
                "credits_used": round(credits_used, 2),
                "cost_breakdown": {
                  "base_processing": self.BASE_COST,
                  "letterboxing": self.LETTERBOX_COST if processing_metadata['letterboxed'] else 0,
                  "tta_steps": self.TTA_STEPS * self.COST_PER_TTA_STEP
                }
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _apply_tta_augmentation(self, img_array):
        """Apply random augmentations for test-time augmentation"""
        return tf.image.random_brightness(
            tf.image.random_contrast(img_array, lower=0.8, upper=1.2),
            max_delta=0.1
        )

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = PathogenPredictor(model_path="./V2B2/model.keras")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on application startup"""
    try:
        # ASCII art and system info
        print(r"""
              _|_|                        _|      _|  _|            _|
            _|    _|  _|_|_|      _|_|    _|      _|        _|_|_|        _|_|    _|_|_|
            _|    _|  _|    _|  _|_|_|_|  _|      _|  _|  _|_|      _|  _|    _|  _|    _|
            _|    _|  _|    _|  _|          _|  _|    _|      _|_|  _|  _|    _|  _|    _|
              _|_|    _|    _|    _|_|_|      _|      _|  _|_|_|    _|    _|_|    _|    _|
        """)
        print(f"""\033[92m
        🌱 OneVision - Advanced Plant Pathogen Detection System 🌿
        Copyright (c) {datetime.datetime.now().year} Fabian Franco-Roldan

        Model Architecture:
        - Backbone: EfficientNetV2B2 (ImageNet pretrained)
        - Custom Head:
          • Global Average Pooling
          • Dense (256 units) with SWISH activation
          • Batch Normalization + Dropout
          • Softmax Classification Layer

        Key Features:
        ✅ 300x300px RGB input resolution
        ✅ Supports 50-1000 plant disease classes
        ✅ Test-Time Augmentation (TTA) for robust predictions
        ✅ EfficientNetV2B2 preprocessing for input normalization
        \033[0m
        """)

        # Load the ML model
        predictor.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model initialization failed")

@app.post("/predict", response_class=JSONResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint for plant pathogen detection"""
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Process image
        contents = await file.read()
        result = predictor.predict(contents)
        return result

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_class=JSONResponse)
async def get_model_info():
    """Dynamically extracts model information including all class labels"""
    try:
        if not predictor.model or not predictor.class_labels:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Find base model through layer inspection
        base_model = None
        for layer in predictor.model.layers:
            if isinstance(layer, tf.keras.Model):
                if 'efficientnetv2-b2' in layer.name.lower():
                    base_model = layer
                    break

        if not base_model:
            raise HTTPException(status_code=500,
                              detail="Could not identify base model architecture")

        # Load class names from the JSON file
        with open(f"{predictor.model_path}_class_names.json", 'r') as f:
            class_names = json.load(f)

        # Build response with class information
        info = {
            "input_specification": {
                "shape": predictor.model.input_shape[1:],
                "dtype": "float32",
                "normalization": {
                    "formula": "(x / 127.5) - 1.0",
                    "expected_range": "[-1.0, 1.0]"
                }
            },
            "class_information": {
                "total_classes": len(class_names),
                "all_classes": class_names,
                "class_format": "plant_disease_variant"
            },
            "base_model": {
                "name": base_model.name,
                "trainable_layers": sum(l.trainable for l in base_model.layers),
                "total_parameters": base_model.count_params()
            },
            "prediction_head": {
                "layers": [
                    {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                        "parameters": layer.count_params()
                    }
                    for layer in predictor.model.layers
                    if layer != base_model
                ]
            },
            "pricing_model": {
                 "base_processing_cost": predictor.BASE_COST,
                 "letterboxing_cost": predictor.LETTERBOX_COST,
                 "cost_per_tta_step": predictor.COST_PER_TTA_STEP,
                 "example_calculation": (
                   f"BASE ({predictor.BASE_COST}) + "
                   f"LETTERBOX ({predictor.LETTERBOX_COST} if needed) + "
                   f"TTA ({predictor.TTA_STEPS} steps × {predictor.COST_PER_TTA_STEP})"
                   )
            }
        }

        return JSONResponse(content=info)

    except Exception as e:
        logger.error(f"Model info error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))