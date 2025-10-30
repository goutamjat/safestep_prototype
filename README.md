SafeStep (Software-only Prototype)

Indoor navigation aid prototype using YOLOv8 object detection on a live webcam feed to produce simple navigation instructions (forward / veer left / veer right / stop) based on obstacle layout.

Features
- Live webcam capture (OpenCV)
- YOLOv8 object detection (Ultralytics)
- Region-based occupancy mapping (left / center / right)
- Simple indoor navigation policy with tunable thresholds
- Optional visualization overlay (bounding boxes, regions, and current instruction)

Installation
1) Python 3.9+ recommended.
2) Install dependencies:
```
pip install -r requirements.txt
```
Notes:
- On Windows, if OpenCV wheels fail, install prebuilt: `pip install opencv-python`.
- Ultralytics will auto-download the YOLO model on first run (default is `yolov8n.pt`).

Usage
Run the prototype:
```
python app.py
```
Keys:
- v: toggle visualization overlays
- c: switch to next camera index
- z: switch to previous camera index
- q or ESC: quit

How it works
- Each frame is split into three vertical regions: left, center, right.
- Detected obstacles (people, chairs, tables, backpacks, etc.) mark regions as occupied if bounding boxes overlap significantly.
- A simple policy suggests a direction:
  - If center is free: Move forward
  - Else prefer the freer side: Veer left or Veer right
  - If all regions blocked: Stop

You can tune:
- Confidence threshold
- Intersection fraction to consider a region blocked
- Classes considered obstacles (see `navigation.py`)

File structure
- app.py: Program entrypoint; webcam loop and UI.
- detector.py: YOLOv8 detection wrapper.
- navigation.py: Region mapping and instruction policy.
- requirements.txt: Python dependencies.

Troubleshooting
- To change cameras at runtime, press 'c' or 'z'.
- If no camera opens, try launching with a specific index: `python app.py --camera 1`.
- If FPS is low, switch to `yolov8n.pt` (nano) or reduce frame size.
- If you get CUDA errors, run on CPU by forcing `device='cpu'` in `detector.py`.

Disclaimer
This is a software-only prototype for indoor navigation research and is not a safety-certified mobility aid. Do not rely on it in hazardous environments.


