"""
LatentBoostClassifier: A hybrid generative model combining CVAE, CGAN, and Random Forest for classification tasks.

Modules:
    - model: Defines the CVAE, CGAN, and Random Forest models and their integrations.
    - train: Handles the training of the hybrid model.
    - visualize: Provides tools to visualize model performance and results.
    - utils: Utility functions such as sampling and hyperparameter tuning.

Author:
    Ali Bavarchee - ali.bavarchee@gmail.com
"""

from .model import build_cvae, build_cgan, tune_cvae, tune_cgan, tune_random_forest, CustomLossLayer, sampling
from .train import parallel_train
from .visualize import visualize_hybrid_model
#from .utils import sampling

__all__ = [
    "build_cvae",
    "build_cgan",
    "tune_cvae",
    "tune_cgan",
    "tune_random_forest",
    "CustomLossLayer",
    "parallel_train",
    "visualize_hybrid_model",
    "sampling"
]