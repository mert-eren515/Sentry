import onnxruntime as ort
ort.preload_dlls()

import cv2
from face_detector import FaceDetector
from recognizer import Recognizer, Voter

detector = FaceDetector()
recognizer = Recognizer()
voter = Voter()
video = cv2.VideoCapture(0)

while True:
    ret, image = video.read()

    faces = detector.detect(image)

    if faces:
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]))
        name, score = recognizer.identify(face)
        stable = voter.update(name)

        x1, y1, x2, y2 = face.bbox.astype(int)
        color = (0, 255, 0) if stable else (0, 0, 255)
        label = f"{stable} {score:.2f}" if stable else "Unknown"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        voter.update(None)

    cv2.imshow("Sentry", image)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
