import gpu
gpu.enable()

import time
import cv2
from face_detector import FaceDetector
from pose_detector import PoseDetector
from recognizer import Recognizer, Voter
from binder import IdentityBinder, match_owner
from gestures import GestureWatcher
from permissions import Permissions

WINDOW = "Sentry"
COLORS = {"OK": (0, 255, 0), "WARN": (0, 165, 255), "ALARM": (0, 0, 255)}
GREEN, RED = (0, 255, 0), (0, 0, 255)
BANNER_SECONDS = 2.0

detector = FaceDetector()
pose = PoseDetector()
recognizer = Recognizer()
voter = Voter()
binder = IdentityBinder()
watcher = GestureWatcher()
rights = Permissions()
video = cv2.VideoCapture(0)


def on_gesture(name, gesture, granted):
    # Where the sentry would actually do something. Printing is the placeholder.
    print(f"{name}: {gesture} -> {'GRANTED' if granted else 'DENIED'}")


last = time.monotonic()
fps = 0.0
banner, banner_color, banner_until = "", GREEN, 0.0

while True:
    ret, image = video.read()
    if not ret:
        break

    faces = detector.detect(image)
    people = pose.detect(image)

    # Still a single-person setup on the face side: the largest face is the one
    # we try to identify. The pose model is what makes everyone else visible.
    face, name, face_width = None, None, 0.0
    if faces:
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]))
        name, score = recognizer.identify(face)
        face_width = float(face.bbox[2] - face.bbox[0])

    identity = voter.update(name)

    # "No face in frame" and "a face that does not match anyone" are different
    # events: the first means the person walked away, the second means a
    # stranger is standing there. The binder gets both, separately.
    owner = match_owner(people, face.bbox) if face is not None and identity else None
    level, message = binder.update(identity, bool(faces), owner, face_width, people)

    # Gestures are only read once the identity is verified, so an unrecognised
    # person waving their arms cannot trigger anything.
    verified = binder.state == binder.BOUND and owner is not None
    gesture, fired = watcher.update(owner if verified else None)
    if fired:
        granted = rights.allowed(binder.identity, gesture)
        on_gesture(binder.identity, gesture, granted)
        banner = f"{gesture}: {'GRANTED' if granted else 'DENIED'}"
        banner_color = GREEN if granted else RED
        banner_until = time.monotonic() + BANNER_SECONDS

    color = COLORS[level]
    pose.draw(image, people, owner)
    if face is not None:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, f"{binder.state}  {fps:.1f} fps  [q] quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if gesture:
        cv2.putText(image, gesture, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
    if time.monotonic() < banner_until:
        cv2.putText(image, banner, (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2)

    cv2.imshow(WINDOW, image)

    # waitKey returns more than the character code on some platforms, so the key
    # has to be masked before comparing. Escape works too, and closing the
    # window with its X button exits as well - the keyboard only reaches us
    # while the window has focus, which is easy to lose.
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

    now = time.monotonic()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
    last = now

video.release()
cv2.destroyAllWindows()
