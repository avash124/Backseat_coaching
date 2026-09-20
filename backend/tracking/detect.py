"""Load YOLO once; emit player-only Nx6 [xyxy, confidence, class_id] arrays."""
from pathlib import Path

import numpy as np

from tracking.config import DetectorConfig


class PlayerDetector:
    def __init__(self, weights: Path, config: DetectorConfig):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Trained weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.config = config
        ids = [index for index, name in self.model.names.items() if name == "player"]
        if len(ids) != 1:
            raise ValueError(f"Expected a 'player' class; weights contain {self.model.names}")
        self.player_id = ids[0]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            source=frame,
            classes=[self.player_id],
            imgsz=self.config.imgsz,
            conf=self.config.confidence,
            max_det=self.config.max_detections,
            device=self.config.device,
            nms=False,
            verbose=False,
        )[0]
        boxes = result.boxes
        return (
            np.column_stack(
                (
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                )
            )
            .astype(np.float32)
            .reshape(-1, 6)
        )
