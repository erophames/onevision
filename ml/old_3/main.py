from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import io
import logging
import re
from PIL import Image, ImageFilter, ImageOps
import base64
import datetime
# This is a sample service that the ruby predictor can access
# An attempt was made to use PyCall bridge.
# But I valued my sanity and decided that with FastAPI the ML "Service"
# could actually be scaled horizontally across various cluster instances
# Then an API load balancing gateway could be placed in front to divert
# traffic on high load scenarios
# - Fabian

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
    """Core prediction functionality extracted from the original class"""
    IMG_SIZE = 300
    TTA_STEPS = 5  # Test-time augmentation steps

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_labels = None

    def load_model(self):
        """Load the trained model and class labels"""
        try:
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    'SparsePrecision': SparsePrecision,
                    'SparseRecall': SparseRecall
                }
            )

            logger.info(self.model.summary())

            # Load class labels
            with open(f"{self.model_path}_class_names.json", 'r') as f:
                self.class_labels = json.load(f)
            logger.info("Model and labels loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_path):
        try:
            # Load model and class labels
            model = keras.models.load_model(self.output_model_path)
            with open(f"{self.output_model_path}_class_names.json", 'r') as f:
                class_names = json.load(f)

            # Load and preprocess image
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))

            img = ImageOps.exif_transpose(img)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

             # Smart image scaling with aspect ratio preservation
            aspect_ratio = img.width / img.height
            if aspect_ratio > 2 or aspect_ratio < 0.5:
                logger.info("Image has extreme ratios - letterboxing")
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

            # Convert the processed image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Apply preprocessing (use the correct preprocessing method)
            img_array = preprocess_input(img_array)

            # Test-time augmentation
            predictions = []
            for _ in range(self.TTA_STEPS):
                augmented = self._apply_tta_augmentation(img_array)
                predictions.append(model.predict(augmented, verbose=0))

            avg_prediction = np.mean(predictions, axis=0)
            conf = np.max(avg_prediction)
            class_idx = np.argmax(avg_prediction)
            class_label = class_names[class_idx]

            logger.info(f"Class Label: {class_label}")
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"

            fruit, disease = re.match(pattern, class_label).groups()
            fruit = re.sub(r"_", " ", fruit)
            disease = re.sub(r"_", " ", disease).title()

            return {
                "fruit": fruit,
                "disease": disease,
                "confidence": float(conf)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predictt(self, image_bytes):
        """Make a prediction on the uploaded image and return the processed image"""
        try:
            # Load and preprocess image
            img = keras.preprocessing.image.load_img(
                io.BytesIO(image_bytes),
                target_size=(self.IMG_SIZE, self.IMG_SIZE)
            )
            img = ImageOps.exif_transpose(img)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

             # Smart image scaling with aspect ratio preservation
            aspect_ratio = img.width / img.height
            if aspect_ratio > 2 or aspect_ratio < 0.5:
                logger.info("Image has extreme ratios - letterboxing")
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

            # Convert the processed image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Perform test-time augmentation
            predictions = []
            for _ in range(self.TTA_STEPS):
                augmented = self._apply_tta_augmentation(img_array)
                predictions.append(self.model.predict(augmented, verbose=0))

            # Average predictions and format results
            avg_prediction = np.mean(predictions, axis=0)
            conf = np.max(avg_prediction)
            class_idx = np.argmax(avg_prediction)
            class_label = self.class_labels[class_idx]

            logger.info(f"Class Label: {class_label}")
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"

            fruit, disease = re.match(pattern, class_label).groups()
            fruit = re.sub(r"_", " ", fruit)
            disease = re.sub(r"_", " ", disease).title()

            return {
                "plant": fruit,
                "disease": disease,
                "confidence": float(conf),
                "processed_image": img_base64  # Include the base64-encoded image
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


    def _apply_tta_augmentation(self, img_array):
        """Apply random augmentations for test-time augmentation (no flipping)"""
        return tf.image.random_brightness(
            tf.image.random_contrast(img_array, lower=0.8, upper=1.2),
            max_delta=0.1
        )

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = PathogenPredictor(model_path="./V2B2/model.keras")

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    try:
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

            Future Plans:

            Reinforcement Learning Strategy (Not Implemented)
            -------------------------------------------------

            ✅ Only updates existing weights through additional training steps
            ✅ Doesn't reinitialize or rebuild the architecture
            ✅ Modifies the original model file over time
            ✅ Maintains historical examples to preserve old knowledge

            Original Model Architecture
                   │
                   ▼
            [Base (EfficientNet)] → [Custom Head]
                   ▲           │
                   └───────────┘
               Continuous Updates via:
               - Farmer feedback samples
               - New uploaded images
               - Regularized fine-tuning

            Purpose:
            Automated detection of plant diseases from leaf images with
            state-of-the-art deep learning for precision agriculture.
            \033[0m
        """)

        predictor.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model initialization failed")

@app.post("/predict", response_class=JSONResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint for processing plant pathogen predictions"""
    try:
        # Read and validate image file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")

        contents = await file.read()
        result = predictor.predict(contents)
        return result

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
