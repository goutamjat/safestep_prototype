import argparse
from typing import Tuple, List

import cv2
import numpy as np

from detector import YoloV8Detector
from navigation import compute_region_occupancy, decide_navigation_instruction


def draw_overlay(
	frame: np.ndarray,
	detections: List[Tuple[int, int, int, int, float, int, str]],
	instruction: str,
	show_boxes: bool = True,
) -> np.ndarray:
	h, w = frame.shape[:2]
	region_width = w // 3
	colors = {
		"left": (64, 64, 64),
		"center": (64, 64, 64),
		"right": (64, 64, 64),
	}
	# Regions grid
	for i in range(1, 3):
		cv2.line(frame, (region_width * i, 0), (region_width * i, h), (80, 80, 80), 1)
	# Detections
	if show_boxes:
		for x1, y1, x2, y2, conf, cid, cname in detections:
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
			label = f"{cname}:{conf:.2f}"
			cv2.putText(frame, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
	# Instruction banner
	cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
	cv2.putText(frame, f"Instruction: {instruction}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
	return frame


def main() -> None:
	parser = argparse.ArgumentParser(description="SafeStep indoor navigation prototype")
	parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model name or path")
	parser.add_argument("--camera", type=int, default=0, help="Webcam index")
	parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
	parser.add_argument("--no-viz", action="store_true", help="Disable visualization overlays")
	parser.add_argument("--width", type=int, default=960, help="Resize capture width")
	parser.add_argument("--height", type=int, default=540, help="Resize capture height")
	args = parser.parse_args()

	detector = YoloV8Detector(model_name=args.model, device=None, conf_threshold=args.conf)
	capture = cv2.VideoCapture(args.camera)
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

	obstacle_classes = {
		"person": True,
		"chair": True,
		"bench": True,
		"sofa": True,
		"bed": True,
		"dining table": True,
		"tv": True,
		"laptop": True,
		"backpack": True,
		"handbag": True,
		"suitcase": True,
		"bottle": True,
		"cup": True,
		"book": True,
		"potted plant": True,
	}

	show_viz = not args.no_viz
	print("Press 'v' to toggle visualization, 'q' or ESC to quit.")

	while True:
		ok, frame = capture.read()
		if not ok:
			print("Camera read failed. Exiting.")
			break
		frame = cv2.resize(frame, (args.width, args.height))
		dets = detector.infer(frame)
		occ = compute_region_occupancy(
			frame_width=frame.shape[1],
			frame_height=frame.shape[0],
			detections=dets,
			obstacle_class_names=obstacle_classes,
			min_horizontal_intersection=0.33,
		)
		instruction = decide_navigation_instruction(occ)
		print(instruction)
		if show_viz:
			vis = draw_overlay(frame.copy(), dets, instruction, show_boxes=True)
			cv2.imshow("SafeStep Prototype", vis)
		key = cv2.waitKey(1) & 0xFF
		if key in (27, ord('q')):
			break
		if key in (ord('v'),):
			show_viz = not show_viz

	capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


