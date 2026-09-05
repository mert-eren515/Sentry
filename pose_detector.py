import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Ultralytics writes settings.json to the user config dir by default
# (AppData\Roaming\Ultralytics). Point it inside the project instead so the
# repo stays self-contained and nothing leaks outside the project folder. The
# directory must already exist, otherwise ultralytics decides it is not
# writable and silently drops the file into the current working directory.
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR))

import cv2
from ultralytics import YOLO

# COCO-17 keypoint indices. We only care about the head and the arm chain.
NOSE = 0
L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

# shoulder -> elbow -> wrist, per side. This is the "chain": a wrist is only
# trusted when the model attached it to a specific person's skeleton.
ARMS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST),
        "right": (R_SHOULDER, R_ELBOW, R_WRIST)}


def _pt(p):
    # cv2 refuses numpy integer tuples on some builds.
    return (int(p[0]), int(p[1]))


class Person:
    # One detected skeleton. kp is (17, 2) pixel coords, kp_score is (17,).
    def __init__(self, box, kp, kp_score, min_score=0.5):
        self.box = box
        self.kp = kp
        self.kp_score = kp_score
        self.min_score = min_score

    def point(self, idx):
        # None means "the model is not confident this joint is visible".
        if self.kp_score[idx] < self.min_score:
            return None
        return self.kp[idx]

    def head(self):
        # Nose is the most stable head point, eyes are the fallback when the
        # head is turned far enough that the nose drops below threshold.
        for idx in (NOSE, L_EYE, R_EYE):
            p = self.point(idx)
            if p is not None:
                return p
        return None

    def wrists(self):
        # Only wrists whose whole arm chain is visible count. A bare wrist with
        # no shoulder/elbow behind it is exactly the case we do not trust.
        found = {}
        for side, (shoulder, elbow, wrist) in ARMS.items():
            if all(self.point(i) is not None for i in (shoulder, elbow, wrist)):
                found[side] = self.kp[wrist]
        return found

    def loose_wrists(self):
        # Visible wrists regardless of the chain, used to spot arms that are
        # reaching in without the body being properly resolved.
        return {side: self.kp[w] for side, (_, _, w) in ARMS.items()
                if self.point(w) is not None}


class PoseDetector:
    def __init__(self, weights="yolo11s-pose.pt", conf=0.5, kp_score=0.5):
        import torch
        self.conf = conf
        self.kp_score = kp_score
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(MODELS_DIR / weights))

    def detect(self, image):
        result = self.model.predict(image, conf=self.conf, device=self.device,
                                    verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        coords = result.keypoints.xy.cpu().numpy()
        scores = result.keypoints.conf
        scores = scores.cpu().numpy() if scores is not None else None

        people = []
        for i in range(len(coords)):
            s = scores[i] if scores is not None else [1.0] * len(coords[i])
            people.append(Person(boxes[i], coords[i], s, self.kp_score))
        return people

    # main.py uses this, mirroring FaceDetector.draw
    def draw(self, image, people, owner=None):
        for person in people:
            color = (0, 255, 0) if person is owner else (0, 165, 255)
            for shoulder, elbow, wrist in ARMS.values():
                a, b, c = (person.point(shoulder), person.point(elbow), person.point(wrist))
                if a is not None and b is not None:
                    cv2.line(image, _pt(a), _pt(b), color, 2)
                if b is not None and c is not None:
                    cv2.line(image, _pt(b), _pt(c), color, 2)
                if c is not None:
                    cv2.circle(image, _pt(c), 6, color, -1)
