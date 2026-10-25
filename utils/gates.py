import numpy as np

from utils.config import CHECK_EVERY, CLOSED_KEEP, EMPTY_FINGER_SUM


class HorizonGate:
    """Shorten a base action queue. Checks land every CHECK_EVERY executed steps.

    gripper: once CLOSED_KEEP actions have commanded a close, drop the rest.
    fingers: two checks in a row with a close command and shut fingers drop the rest.
    both: either condition drops the rest.
    """

    def __init__(self, mode):
        if mode not in ("gripper", "fingers", "both"):
            raise ValueError(f"unknown gate {mode}")
        self.mode = mode
        self.age = 0
        self.closed_age = 0
        self.empty_hits = 0

    def reset_chunk(self):
        self.age = 0
        self.closed_age = 0
        self.empty_hits = 0

    def after_step(self, queue, action, finger_qpos):
        grip = float(np.asarray(action).reshape(-1)[-1])
        opening = float(np.abs(np.asarray(finger_qpos)).sum())
        self.age += 1
        if grip > 0.0:
            self.closed_age += 1
        else:
            self.closed_age = 0
            self.empty_hits = 0
        if self.age % CHECK_EVERY != 0:
            return
        cut = False
        if self.mode in ("gripper", "both") and self.closed_age >= CLOSED_KEEP:
            cut = True
        if self.mode in ("fingers", "both") and grip > 0.0:
            if opening < EMPTY_FINGER_SUM:
                self.empty_hits += 1
            else:
                self.empty_hits = 0
            if self.empty_hits >= 2:
                cut = True
        if cut:
            queue.clear()
            self.reset_chunk()
