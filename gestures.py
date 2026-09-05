import numpy as np

from recognizer import Voter
from pose_detector import (NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST,
                           L_ELBOW, R_ELBOW)

# Every measurement below is divided by the shoulder width. That is the same
# trick the reach test in binder.py uses: it makes the thresholds independent of
# how far the person is standing from the camera.
#
# Note these are arm poses, not finger poses. The pose model reports a wrist as
# a single point and knows nothing about fingers, so "open palm" or "thumbs up"
# are out of reach without a second, hand-specific model.


def _scale(person):
    ls, rs = person.point(L_SHOULDER), person.point(R_SHOULDER)
    if ls is None or rs is None:
        return None, None, None
    width = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
    if width < 1.0:
        return None, None, None
    midline = (ls[0] + rs[0]) / 2.0
    return width, midline, (ls, rs)


def hands_up(person):
    # Both wrists above the head. Remember y grows downwards in an image.
    nose = person.point(NOSE)
    lw, rw = person.point(L_WRIST), person.point(R_WRIST)
    if nose is None or lw is None or rw is None:
        return False
    return bool(lw[1] < nose[1] and rw[1] < nose[1])


def _one_arm_raised(person, up_wrist, down_wrist, down_shoulder):
    nose = person.point(NOSE)
    up, down = person.point(up_wrist), person.point(down_wrist)
    shoulder = person.point(down_shoulder)
    if nose is None or up is None or down is None or shoulder is None:
        return False
    # One hand clearly above the head while the other stays down, so this never
    # fires as a half-detected hands_up.
    return bool(up[1] < nose[1] and down[1] > shoulder[1])


def left_arm_raised(person):
    # "Left" is the person's own left, which appears on the right of the frame.
    return _one_arm_raised(person, L_WRIST, R_WRIST, R_SHOULDER)


def right_arm_raised(person):
    return _one_arm_raised(person, R_WRIST, L_WRIST, L_SHOULDER)


def arms_crossed(person):
    width, midline, shoulders = _scale(person)
    if width is None:
        return False
    ls, rs = shoulders
    lw, rw = person.point(L_WRIST), person.point(R_WRIST)
    if lw is None or rw is None:
        return False

    # Each wrist has to sit on the far side of the body midline from its own
    # shoulder. Comparing signs rather than raw x keeps this working whichever
    # way round the person is standing.
    crossed = (np.sign(lw[0] - midline) != np.sign(ls[0] - midline) and
               np.sign(rw[0] - midline) != np.sign(rs[0] - midline))

    # Chest height: below the shoulders but not down by the hips.
    top = min(ls[1], rs[1])
    chest = top < lw[1] < top + 1.5 * width and top < rw[1] < top + 1.5 * width
    return bool(crossed and chest)


def t_pose(person):
    width, midline, shoulders = _scale(person)
    if width is None:
        return False
    ls, rs = shoulders
    lw, rw = person.point(L_WRIST), person.point(R_WRIST)
    if lw is None or rw is None:
        return False
    level = (abs(lw[1] - ls[1]) < 0.4 * width and abs(rw[1] - rs[1]) < 0.4 * width)
    spread = (abs(lw[0] - midline) > 1.2 * width and abs(rw[0] - midline) > 1.2 * width)
    return bool(level and spread)


# Order matters: the first stable match wins, so the more specific poses are
# checked before the looser ones.
GESTURES = {
    "t_pose": t_pose,
    "arms_crossed": arms_crossed,
    "hands_up": hands_up,
    "left_arm_raised": left_arm_raised,
    "right_arm_raised": right_arm_raised,
}


class GestureWatcher:
    # A gesture has to hold for a few frames before it counts, and it fires once
    # rather than on every frame it stays up. Without the latch a raised hand
    # would trigger thirty times a second.
    def __init__(self):
        self.voters = {name: Voter() for name in GESTURES}
        self.active = None

    def update(self, person):
        stable = None
        for name, test in GESTURES.items():
            # bool() is deliberate: a numpy bool here would never match `is True`.
            hit = self.voters[name].update(bool(person is not None and test(person)))
            if stable is None and hit is True:
                stable = name

        fired = stable is not None and stable != self.active
        self.active = stable
        return stable, fired
