from functools import lru_cache

from ultralytics import YOLO


@lru_cache(maxsize=1)
def get_models() -> dict:
    """
    Load and return the YOLO models used for inference.

    This function is cached so models are loaded only once per process.
    """
    model_v8 = YOLO("yolov8n.pt")
    model_v12 = YOLO("yolov12.pt")

    return {
        "yolov8n": model_v8,
        "yolov12": model_v12,
    }

