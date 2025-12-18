"""
AI Package - Advanced Model Management für Overnight Development.

Zentrale AI-Intelligenz mit State-of-the-Art Models:
✅ Smart Model Selection basierend auf Hardware
✅ Performance-Optimized Models (ONNX, TensorRT)
✅ Quality vs Speed Trade-offs
✅ Hardware-Aware Model Loading
✅ Advanced CLIP, BLIP-2, LLaVA Models
"""

from .model_manager import (
    ModelFramework,
    ModelManager,
    ModelPrecision,
    ModelRegistry,
    ModelSpec,
    ModelType,
    SmartModelSelector,
    get_model_manager,
    get_optimal_models_for_hardware,
)

__all__ = [
    "ModelManager",
    "ModelType",
    "ModelFramework",
    "ModelPrecision",
    "ModelSpec",
    "ModelRegistry",
    "SmartModelSelector",
    "get_model_manager",
    "get_optimal_models_for_hardware",
]
