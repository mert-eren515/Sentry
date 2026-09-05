import time
import numpy as np

from recognizer import Voter

# A face is roughly 15 cm wide and the distance from the head to a fully
# extended hand is roughly 85 cm, so a wrist further than ~6 face widths from
# the head cannot belong to that head. Using the face box as the unit makes
# this scale invariant: step back from the camera and both shrink together.
MAX_REACH = 6.0


def match_owner(people, face_bbox, margin=0.25):
    # Decide which detected skeleton belongs to the recognised face. The head
    # keypoint has to fall inside the face box; if several do, take the closest
    # one to the box centre. The box is grown a little first, because the two
    # models disagree slightly on where a head is and a tight box drops matches
    # as soon as the person steps back from the camera.
    x1, y1, x2, y2 = face_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    best, best_dist = None, None
    for person in people:
        head = person.head()
        if head is None:
            continue
        if not (x1 <= head[0] <= x2 and y1 <= head[1] <= y2):
            continue
        dist = np.hypot(head[0] - cx, head[1] - cy)
        if best_dist is None or dist < best_dist:
            best, best_dist = person, dist
    return best


def in_reach(wrist, head, face_width):
    return np.hypot(wrist[0] - head[0], wrist[1] - head[1]) <= MAX_REACH * face_width


def owner_hands(owner, face_width):
    # Chained wrists that are also anatomically plausible. The reach test is a
    # second line of defence for the case where the pose model merges two
    # overlapping people into one skeleton.
    head = owner.head()
    if head is None:
        return {}
    return {side: w for side, w in owner.wrists().items()
            if in_reach(w, head, face_width)}


# A MediaPipe hand and a pose wrist are the same joint seen by two models, so
# they land close together. Anything further apart than this - measured in face
# widths, like the reach test - is a different hand.
HAND_MATCH = 1.5


def match_hands(hands, wrists, face_width):
    # Split detected hands into the ones sitting on a wrist we already trust and
    # the ones we cannot account for. The second group is what closes the blind
    # spot: the pose model never reports a bare forearm as a person, but the
    # hand model sees the hand, and that hand belongs to nobody.
    owned, loose = [], []
    limit = HAND_MATCH * face_width
    for hand in hands:
        near = [np.hypot(hand.wrist[0] - w[0], hand.wrist[1] - w[1])
                for w in wrists.values()]
        (owned if near and min(near) <= limit else loose).append(hand)
    return owned, loose


def foreign_hands(people, owner):
    # Every wrist in the frame that is not the owner's. This is the threat we
    # care about: a hand reaching in from somewhere else while the known person
    # is standing in front of the camera.
    #
    # Known blind spot: the pose model only reports wrists that it managed to
    # attach to a person it detected. A bare forearm entering the frame with no
    # body behind it may not register as a person at all, and then we never see
    # that wrist. Closing this needs a hand detector running alongside the pose
    # model, which is why detect() is kept separate from the chain logic here.
    count = 0
    for person in people:
        if person is owner:
            continue
        count += len(person.loose_wrists())
    return count


class IdentityBinder:
    IDLE = "IDLE"
    AWAITING_HANDS = "AWAITING_HANDS"
    BOUND = "BOUND"
    WARN = "WARN"

    OK = "OK"
    ALARM = "ALARM"

    def __init__(self, hands_timeout=15.0, lease=5.0):
        # hands_timeout: how long we wait for the person to show both hands.
        # lease: how long the identity survives after the face leaves the frame.
        self.hands_timeout = hands_timeout
        self.lease = lease

        self.state = self.IDLE
        self.identity = None
        self.since = time.monotonic()

        # Single frames lie: a hand blurs, a wrist drops below threshold for one
        # frame. Same majority-vote trick the face recogniser already uses.
        self.owned_hands = []
        self.hands_voter = Voter()
        self.intruder_voter = Voter()
        self.unknown_voter = Voter()

    def _enter(self, state):
        if state != self.state:
            self.state = state
            self.since = time.monotonic()

    def elapsed(self):
        return time.monotonic() - self.since

    def update(self, identity, face_seen, owner, face_width, people, hands=()):
        # identity: stable name from the face Voter, None when not recognised
        # face_seen: a face was detected at all, recognised or not
        wrists = owner_hands(owner, face_width) if owner is not None else {}
        both_hands = self.hands_voter.update(len(wrists) >= 2) is True

        # Hands the owner cannot account for count as intruders too, alongside
        # whole people the pose model resolved separately.
        self.owned_hands, loose = match_hands(hands, wrists, face_width or 1.0)
        strangers = foreign_hands(people, owner) > 0 or (owner is not None and loose)
        intruders = self.intruder_voter.update(bool(strangers)) is True
        unknown = self.unknown_voter.update(face_seen and identity is None) is True

        if identity is not None:
            self.identity = identity

        # The face is recognised but the pose model found no skeleton to hang it
        # on (person too close, body out of frame). We cannot judge any hand in
        # this state, so we must not claim the ones we see are intruders.
        no_chain = identity is not None and owner is None

        if self.state == self.IDLE:
            if identity is not None:
                self._enter(self.AWAITING_HANDS)

        elif self.state == self.AWAITING_HANDS:
            if identity is None:
                self._enter(self.IDLE)
                self.identity = None
            elif both_hands:
                self._enter(self.BOUND)
            elif self.elapsed() > self.hands_timeout:
                # Refused to show hands. Drop back and make them start over.
                self._enter(self.IDLE)
                self.identity = None

        elif self.state == self.BOUND:
            # The identity now lives on the face, so the hands are free to move
            # out of frame. Only losing the face breaks the binding.
            if identity is None:
                self._enter(self.WARN)

        elif self.state == self.WARN:
            if identity is not None:
                self._enter(self.BOUND)
            elif self.elapsed() > self.lease:
                # Face gone too long. We cannot prove the person in front of the
                # camera is still the same one, so the identity has to go.
                self._enter(self.IDLE)
                self.identity = None

        return self._report(unknown, intruders, no_chain, len(wrists))

    def _report(self, unknown, intruders, no_chain, hand_count):
        if unknown:
            return self.ALARM, "UNKNOWN PERSON"
        if no_chain:
            return self.WARN, f"{self.identity}: NO BODY CHAIN"
        # Only judged while the face is present. In WARN the person has already
        # left, so their own lingering hands would fire this on every turn away.
        if intruders and self.state in (self.BOUND, self.AWAITING_HANDS):
            return self.ALARM, "UNCHAINED HAND"
        if self.state == self.WARN:
            return self.WARN, f"{self.identity} LEFT FRAME"
        if self.state == self.AWAITING_HANDS:
            return self.OK, f"{self.identity}: SHOW BOTH HANDS ({hand_count}/2)"
        if self.state == self.BOUND:
            return self.OK, f"{self.identity} VERIFIED"
        return self.OK, "NO IDENTITY"
