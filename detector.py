from typing import List, Tuple, Optional

import numpy as np

try:
	from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
	YOLO = None  # type: ignore


class YoloV8Detector:
	"""Thin wrapper around Ultralytics YOLOv8 for webcam frames.

	Returns detections as list of tuples: (x1, y1, x2, y2, conf, class_id, class_name)
	All coordinates are in pixel space of the input frame.
	"""

	def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None, conf_threshold: float = 0.35) -> None:
		if YOLO is None:
			raise RuntimeError("Ultralytics YOLO is not available. Please install 'ultralytics'.")
		self.model = YOLO(model_name)
		self.device = device
		self.conf_threshold = conf_threshold

	def infer(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int, str]]:
		results = self.model.predict(source=frame_bgr, verbose=False, device=self.device, conf=self.conf_threshold)
		parsed: List[Tuple[int, int, int, int, float, int, str]] = []
		if not results:
			return parsed
		first = results[0]
		if first is None or first.boxes is None:
			return parsed
		boxes = first.boxes
		xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
		confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
		cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
		names = first.names if hasattr(first, "names") else {}
		for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
			name = names.get(int(cid), str(int(cid)))
			parsed.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cid), str(name)))
		return parsed


