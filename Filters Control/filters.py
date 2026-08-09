import numpy as np
import cv2

def defaultGreyscale(frame: np.ndarray) -> np.ndarray:
    """
    Default greyscale filter.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def greyscale(frame: np.ndarray) -> np.ndarray:
    """
    Custom greyscale formula (matches greyscale() in filters.cpp):
    grey = (blue + green) / 2 - red
    Uses natural uint8 wraparound (matches original C++ behavior), no clipping.
    """
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)

    grey = ((b + g) // 2 - r).astype(np.uint8)

    dst = np.stack([grey, grey, grey], axis=-1)
    return dst