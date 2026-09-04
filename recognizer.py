import numpy as np
from pathlib import Path

class Recognizer:
    def __init__(self, folder="faces", threshold=0.38):
        self.threshold = threshold
        self.names = []
        self.embeddings = []

        for file in Path(folder).glob("*.npy"):
            vectors = np.load(file)
            for v in vectors:
                self.embeddings.append(v)
                self.names.append(file.stem)

        self.embeddings = np.array(self.embeddings)
        print(f"{len(set(self.names))} person, {len(self.names)} vector")

    def identify(self, face):
        if len(self.embeddings) == 0:
            return None, 0.0

        scores = self.embeddings @ face.normed_embedding
        best = scores.argmax()

        if scores[best] > self.threshold:
            return self.names[best], scores[best]
        return None, scores[best]


# Recognition can fail on individual frames due to motion blur or head angle.
# Instead of trusting a single frame, we keep the last 5 results and take the
# majority vote. A name is only accepted if it appears at least 3 times.
# Currently we only track the largest face in the frame (single-person setup).
from collections import deque, Counter

class Voter:
    def __init__(self, window=5, min_votes=3):
        self.window = window
        self.min_votes = min_votes
        self.history = deque(maxlen=window)

    def update(self, name):
        self.history.append(name)
        winner, count = Counter(self.history).most_common(1)[0]
        return winner if count >= self.min_votes else None
