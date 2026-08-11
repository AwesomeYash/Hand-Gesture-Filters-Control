from matplotlib.pyplot import hsv
import numpy as np
import cv2
import mediapipe as mp


# ==============================
# Regular Filters
# ==============================

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
                             [0.393, 0.769, 0.189]], dtype=np.float32)

    # Ensuring 3 channel input
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Converting to float32 for precision during matrix multiplication
    frame_float = frame.astype(np.float32)
    sepia_frame = cv2.transform(frame_float, sepia_kernel)
    dst = np.clip(sepia_frame, 0, 255).astype(np.uint8)

    return dst

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

    b = np.clip(b.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int16) - 10, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.int16) - 40, 0, 255).astype(np.uint8)
    cool_frame = cv2.merge((b, g, r))

    return cool_frame

def warmFilter(frame: np.ndarray) -> np.ndarray:
    """
    Warm Filter
    """
    b, g, r = cv2.split(frame)

    b = np.clip(b.astype(np.int16) - 50, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    warm_frame = cv2.merge((b, g, r))

    return warm_frame

def colorFilter(frame: np.ndarray, color:str) -> np.ndarray:
    """
    Color Highlight Filter
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for highlighting
    lower_bound = np.array([0, 0, 70])
    upper_bound = np.array([0, 0, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    dst = gray_frame_bgr.copy()
    dst[mask != 0] = frame[mask != 0]

    return dst

# ==============================
# Convolution Filters
# ==============================

def blurFilter(frame: np.ndarray) -> np.ndarray:
    """
    Blur Filter
    """
    kernel = np.array([1, 2, 4, 2, 1], dtype=np.float32)
    kernel = kernel / np.sum(kernel)  # Normalize the kernel

    dst = cv2.sepFilter2D(frame, -1, kernel, kernel)

    return dst

def blur5x5(frame: np.ndarray) -> np.ndarray:
    """
    Blur 5x5 Filter
    """
    return cv2.GaussianBlur(frame, (15, 15), 0)

def sobelX(frame: np.ndarray) -> np.ndarray:
    """
    Sobel X Filter
    """
    sobel_x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(sobel_x, cv2.CV_32F, dx=1, dy=0, ksize=3)

    return sobel_x

def sobelY(frame: np.ndarray) -> np.ndarray:
    """
    Sobel Y Filter
    """
    sobel_y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(sobel_y, cv2.CV_32F, dx=0, dy=1, ksize=3)

    return sobel_y

def magnitudeSobel(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """
    Magnitude Sobel Filter
    """
    mag = cv2.magnitude(sx, sy)
    dst = cv2.convertScaleAbs(mag)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst
        
def embossFilter(frame: np.ndarray) -> np.ndarray:
    """
    Emboss Filter
    """
    emboss_kernel = np.array([[ -2, -1, 0],
                              [ -1,  1, 1],
                              [  0,  1, 2]], dtype=np.float32)

    dst = cv2.filter2D(frame, -1, emboss_kernel)
    
    return dst

# ==============================
# Face Recognition Filters
# ==============================

faceDetection = mp.solutions.face_detection.FaceDetection(
                model_selection = 1, 
                min_detection_confidence = 0.6
                )

def faceFinder (frame: np.ndarray) -> np.ndarray:
    """
    Face Finder Function
    """
    dst = frame.copy()
    h, w, _ = frame.shape

    # Convert the frame to RGB for face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            boxWidth = int(bbox.width * w)
            boxHeight = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            boxWidth = min(w - x, boxWidth)
            boxHeight = min(h - y, boxHeight)

            # Draw a rectangle around the detected face
            cv2.rectangle(dst, (x, y), (x + boxWidth, y + boxHeight), # Dimensions
                        (0, 255, 0), 2)   # Color and thickness of BBox

    return dst

