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

def filterSepia(frame: np.ndarray) -> np.ndarray:
    """
    Sepia filter.
    """
    sepia_kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]]), dtype=np.float32)

    # Ensuring 3 channel input
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Converting to float32 for precision during matrix multiplication
    frame_float = frame.astype(np.float32)
    sepia_frame = cv2.transform(frame_float, sepia_kernel)

    return sepia_frame

def filterNegative(frame: np.ndarray) -> np.ndarray:
    """
    Negative filter.
    """
    return 255 - frame

def coolFilter(frame: np.ndarray) -> np.ndarray:
    """
    Cool filter.
    """
    b, g, r = cv2.split(frame)

    b = np.clip(b + 50, 0, 255).astype(np.uint8)
    r = np.clip(r - 50, 0, 255).astype(np.uint8)

    cool_frame = cv2.merge((b, g, r))

    return cool_frame