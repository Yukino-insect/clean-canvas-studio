import cv2
import numpy as np


class OpenCVInpaint:

    def __init__(self):
        pass

    def inpaint(self, frame, mask):
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    def __call__(self, frames: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
        comp = []
        for frame in frames:
            comp.append(self.inpaint(frame, mask))
        return comp