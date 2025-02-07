import argparse
import json
import datetime
from model.model_builder import AdvancedPathogenDetector
from utils.logging_config import setup_logging
from utils.arg_parser import parse_arguments

def main():
    setup_logging()
    args = parse_arguments()

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

        Backbone: EfficientNetV2B2 (ImageNet pretrained)

        Custom Head:
        • Hybrid Attention Block (Channel + Spatial Attention)
        • Global Average Pooling
        • Dense Layers (256 units) with SWISH activation
        • Batch Normalization + Dropout
        • Temperature Scaling for confidence calibration
        • Softmax Classification Layer

        Key Features:
        ✅ 300x300px RGB input resolution
        ✅ Supports 50-1000 plant disease classes
        ✅ Test-Time Augmentation (TTA) for robust predictions
        ✅ Hybrid Attention Mechanism for enhanced feature extraction
        ✅ Temperature Scaling for calibrated confidence scores
        ✅ EfficientNetV2B2 preprocessing for input normalization
        ✅ Class Weighting for imbalanced datasets
        ✅ Hyperparameter Tuning (Random Search, Bayesian Optimization, Hyperband)
        ✅ Custom Metrics: SparseMacroF1, SparseF1, SparsePrecision, SparseRecall
        ✅ Advanced Training Pipeline:

          ✅ Initial training with frozen backbone
          ✅ Fine-tuning with partial unfreezing
          ✅ Temperature calibration for confidence calibration
          ✅ Early stopping, learning rate scheduling, and model checkpointing

        Purpose:
        Automated detection of plant diseases from leaf images using state-of-the-art deep learning techniques for precision agriculture.
        The model incorporates advanced attention mechanisms, confidence calibration, robust data augmentation, and hyperparameter tuning for reliable predictions.
        \033[0m
    """)
    
    detector = AdvancedPathogenDetector(args.dataset, args.output)

    if args.image:
        result = detector.predict(args.image)
        print(json.dumps(result, indent=2))
    elif args.dataset:
        detector.train(tune_hyperparams=args.tune)

if __name__ == "__main__":
    main()
