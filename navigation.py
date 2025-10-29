from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class RegionOccupancy:
	left_blocked: bool
	center_blocked: bool
	right_blocked: bool


def _intersection_fraction(x1: int, y1: int, x2: int, y2: int, rx1: int, ry1: int, rx2: int, ry2: int) -> float:
	ix1 = max(x1, rx1)
	ix2 = min(x2, rx2)
	if ix2 <= ix1:
		return 0.0
	width = float(ix2 - ix1)
	box_width = max(1.0, float(x2 - x1))
	return width / box_width


def compute_region_occupancy(
	frame_width: int,
	frame_height: int,
	detections: List[Tuple[int, int, int, int, float, int, str]],
	obstacle_class_names: Dict[str, bool],
	min_horizontal_intersection: float = 0.33,
) -> RegionOccupancy:
	region_width = frame_width // 3
	regions = {
		"left": (0, 0, region_width, frame_height),
		"center": (region_width, 0, region_width * 2, frame_height),
		"right": (region_width * 2, 0, frame_width, frame_height),
	}
	blocked = {"left": False, "center": False, "right": False}
	for x1, y1, x2, y2, conf, cid, cname in detections:
		if cname not in obstacle_class_names:
			continue
		for key, (rx1, ry1, rx2, ry2) in regions.items():
			frac = _intersection_fraction(x1, y1, x2, y2, rx1, ry1, rx2, ry2)
			if frac >= min_horizontal_intersection:
				blocked[key] = True
	return RegionOccupancy(
		left_blocked=blocked["left"],
		center_blocked=blocked["center"],
		right_blocked=blocked["right"],
	)


def decide_navigation_instruction(occupancy: RegionOccupancy) -> str:
	if not occupancy.center_blocked:
		return "Move forward"
	if occupancy.center_blocked and not occupancy.left_blocked and occupancy.right_blocked:
		return "Turn left"
	if occupancy.center_blocked and occupancy.left_blocked and not occupancy.right_blocked:
		return "Turn right"
	if occupancy.center_blocked and not occupancy.left_blocked and not occupancy.right_blocked:
		return "Choose clearer side"
	return "Stop"


