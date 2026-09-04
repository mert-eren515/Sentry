import cv2
from insightface.app import FaceAnalysis
from pathlib import Path

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', root=Path(__file__).parent, allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image):
        return self.app.get(image)

    # enroll.py uses this this method, not main.py
    def draw(self, image, faces):
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
