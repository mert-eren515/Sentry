import onnxruntime as ort
ort.preload_dlls()

import cv2
import numpy as np
from face_detector import FaceDetector
from pathlib import Path

Path("faces").mkdir(exist_ok=True)

name = input("Name: ")

detector = FaceDetector()
video = cv2.VideoCapture(0)
embeddings = []

print("SPACE = Save, Q = Exit")

while True:
    ret, image = video.read()

    faces = detector.detect(image)
    detector.draw(image, faces)

    cv2.putText(image, f"{len(embeddings)} saved faces", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Save Face", image)

    k = cv2.waitKey(1)
    if k == ord(' '):
        if len(faces) == 1: # If there is 1 face in the frame
            embeddings.append(faces[0].normed_embedding) # Add the face to embeddings as a 512 vector
            print(f"Saved ({len(embeddings)})")
        else:
            print(f"There are {len(faces)} faces, there must be 1 face")
    elif k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

if embeddings:
    np.save(f"faces/{name}.npy", np.array(embeddings))
    print(f"{len(embeddings)} vector saved")
