import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Bones to draw: palm outline plus the four joints of each finger.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _ensure_model():
    MODELS_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        return
    print(f"downloading hand_landmarker.task -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class Hand:
    # 21 landmarks in pixel coordinates. Index 0 is the wrist, which is what
    # ties a hand back to an arm that the pose model has already chained.
    def __init__(self, lm, handedness):
        self.lm = lm
        self.handedness = handedness

    @property
    def wrist(self):
        return self.lm[0]


class HandDetector:
    def __init__(self, max_hands=4, confidence=0.5):
        _ensure_model()
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start = time.monotonic()
        self.last_stamp = -1

    def detect(self, image):
        # MediaPipe wants RGB and a timestamp that never goes backwards.
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Two frames inside the same millisecond would repeat the timestamp,
        # and video mode rejects anything that is not strictly increasing.
        stamp = max(int((time.monotonic() - self.start) * 1000), self.last_stamp + 1)
        self.last_stamp = stamp
        result = self.landmarker.detect_for_video(rgb, stamp)

        height, width = image.shape[:2]
        hands = []
        for marks, handed in zip(result.hand_landmarks, result.handedness):
            points = np.array([[p.x * width, p.y * height] for p in marks])
            # Handedness assumes a mirrored, selfie-style image, so it is not
            # reliable here. Nothing depends on it - hands are tied to people by
            # wrist position, not by which hand MediaPipe thinks it is.
            hands.append(Hand(points, handed[0].category_name))
        return hands

    def draw(self, image, hands, color=(255, 200, 0)):
        for hand in hands:
            for a, b in CONNECTIONS:
                pa = (int(hand.lm[a][0]), int(hand.lm[a][1]))
                pb = (int(hand.lm[b][0]), int(hand.lm[b][1]))
                cv2.line(image, pa, pb, color, 1)
