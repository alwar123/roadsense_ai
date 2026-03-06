import base64
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .db import insert_hazard
from .models_loader import get_models


ROAD_HAZARD_LABELS: Set[str] = {
    "pothole",
    "stone",
    "rock",
    "fallen tree",
    "fallen_tree",
    "fallen-tree",
    "roadblock",
    "road_block",
}


def _decode_image_from_base64(b64_data: str) -> np.ndarray:
    """
    Decode a base64-encoded image (without data URL prefix) into a BGR OpenCV image.
    """
    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image data")
    return img


def _merge_detections(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge detections coming from multiple models by label.

    For each label, keep the detection with the highest confidence.
    """
    best_by_label: Dict[str, Dict[str, Any]] = {}
    for det in detections:
        label = det.get("label")
        if not label:
            continue
        conf = float(det.get("confidence", 0.0))
        existing = best_by_label.get(label)
        if existing is None or conf > float(existing.get("confidence", 0.0)):
            best_by_label[label] = det
    return list(best_by_label.values())


async def websocket_inference_handler(websocket: WebSocket) -> None:
    """
    Handle a WebSocket connection for real-time hazard detection.

    Expected incoming message (JSON text):
    {
        "image": "<base64-encoded image data WITHOUT data URL prefix>",
        "latitude": <float, optional>,
        "longitude": <float, optional>
    }

    Response message (JSON text):
    {
        "detections": [
            {"label": "<class_name>", "confidence": 0.87}
        ],
        "hazards": [
            {
                "hazard_type": "<class_name>",
                "latitude": <float>,
                "longitude": <float>,
                "timestamp": "<ISO 8601 UTC>"
            }
        ]
    }
    """
    await websocket.accept()

    models = get_models()
    model_v8 = models["yolov8n"]
    model_v12 = models["yolov12"]

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON payload"}))
                continue

            image_b64 = payload.get("image")
            latitude = payload.get("latitude")
            longitude = payload.get("longitude")

            if not image_b64:
                await websocket.send_text(json.dumps({"error": "Missing 'image' field"}))
                continue

            try:
                frame = _decode_image_from_base64(image_b64)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_text(
                    json.dumps({"error": f"Failed to decode image: {exc}"})
                )
                continue

            all_detections: List[Dict[str, Any]] = []

            # Run inference with YOLOv8
            results_v8 = model_v8(frame, verbose=False)
            for result in results_v8:
                boxes = result.boxes
                names = result.names
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names.get(cls_id, str(cls_id))
                    all_detections.append(
                        {
                            "label": label,
                            "confidence": round(conf, 4),
                            "model": "yolov8n",
                        }
                    )

            # Run inference with YOLOv12
            results_v12 = model_v12(frame, verbose=False)
            for result in results_v12:
                boxes = result.boxes
                names = result.names
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names.get(cls_id, str(cls_id))
                    all_detections.append(
                        {
                            "label": label,
                            "confidence": round(conf, 4),
                            "model": "yolov12",
                        }
                    )

            merged = _merge_detections(all_detections)

            hazards_to_store: List[Dict[str, Any]] = []
            timestamp = datetime.now(timezone.utc).isoformat()

            if latitude is not None and longitude is not None:
                for det in merged:
                    label = det["label"]
                    normalized_label = label.lower().replace("_", " ").replace("-", " ")
                    if (
                        label.lower() in ROAD_HAZARD_LABELS
                        or normalized_label in ROAD_HAZARD_LABELS
                    ):
                        # Store hazard in DB
                        await insert_hazard(
                            hazard_type=label,
                            latitude=float(latitude),
                            longitude=float(longitude),
                            timestamp=datetime.fromisoformat(timestamp),
                        )
                        hazards_to_store.append(
                            {
                                "hazard_type": label,
                                "latitude": float(latitude),
                                "longitude": float(longitude),
                                "timestamp": timestamp,
                            }
                        )

            response = {
                "detections": merged,
                "hazards": hazards_to_store,
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        # Client disconnected; simply end the handler.
        return

