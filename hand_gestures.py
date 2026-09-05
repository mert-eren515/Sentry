import numpy as np

# MediaPipe hand landmarks. Every finger is four points running outwards from
# the palm, so the tip is always the last one.
WRIST = 0
THUMB = (1, 2, 3, 4)          # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)          # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (INDEX, MIDDLE, RING, PINKY)


def _extended(lm, finger):
    # A finger is straight when its tip has travelled further from the wrist
    # than its middle joint. Curling it brings the tip back towards the palm,
    # which flips this comparison regardless of how the hand is rotated.
    wrist = lm[WRIST]
    tip = np.linalg.norm(lm[finger[3]] - wrist)
    knuckle = np.linalg.norm(lm[finger[1]] - wrist)
    return bool(tip > knuckle)


def _thumb_extended(lm):
    # The thumb folds sideways rather than curling, so its middle joint barely
    # moves. Measuring against the base joint instead gives a usable signal.
    wrist = lm[WRIST]
    return bool(np.linalg.norm(lm[THUMB[3]] - wrist) >
                np.linalg.norm(lm[THUMB[1]] - wrist) * 1.15)


def fingers_up(lm):
    # [thumb, index, middle, ring, pinky] - the building block for everything.
    return [_thumb_extended(lm)] + [_extended(lm, f) for f in FINGERS]


def _hand_size(lm):
    # Wrist to the middle finger's base knuckle: the one distance on a hand that
    # does not change when the fingers move. Used to make thresholds scale free.
    return float(np.linalg.norm(lm[MIDDLE[0]] - lm[WRIST]))


def _thumb_points(lm, sign):
    # sign -1 means up the image, +1 means down. y grows downwards.
    size = _hand_size(lm)
    if size < 1.0:
        return False
    thumb_tip = lm[THUMB[3]]
    # Clearly beyond the wrist in that direction, and beyond the folded fingers
    # too, so a sideways fist cannot pass as a thumbs up.
    past_wrist = (thumb_tip[1] - lm[WRIST][1]) * sign > 0.5 * size
    tips = [lm[f[3]][1] for f in FINGERS]
    past_fingers = all((thumb_tip[1] - t) * sign > 0 for t in tips)
    return bool(past_wrist and past_fingers)


def thumbs_up(hand):
    up = fingers_up(hand.lm)
    return bool(up[0] and not any(up[1:]) and _thumb_points(hand.lm, -1))


def thumbs_down(hand):
    up = fingers_up(hand.lm)
    return bool(up[0] and not any(up[1:]) and _thumb_points(hand.lm, +1))


def fist(hand):
    return not any(fingers_up(hand.lm))


def open_palm(hand):
    return all(fingers_up(hand.lm))


def peace(hand):
    up = fingers_up(hand.lm)
    return bool(up[1] and up[2] and not up[3] and not up[4])


def pointing(hand):
    up = fingers_up(hand.lm)
    return bool(up[1] and not up[2] and not up[3] and not up[4])


# Checked in order, first stable match wins. The specific shapes come before the
# loose ones so that a thumbs up is never reported as a plain fist.
HAND_GESTURES = {
    "thumbs_up": thumbs_up,
    "thumbs_down": thumbs_down,
    "peace": peace,
    "pointing": pointing,
    "open_palm": open_palm,
    "fist": fist,
}
