import cv2
import time
import math
import numpy as np
import mediapipe as mp

from filters import (
    blur5x5, defaultGreyscale, greyscale, filterSepia, filterNegative,
    coolFilter, warmFilter, colorFilter,
    blurFilter, sobelX, sobelY, magnitudeSobel, embossFilter, faceFinder,
    faceBlurFilter
)
WIDTH, HEIGHT = 960, 720

FILTER_NAMES = ["Default Grey", "Blur 15x15", "Cool Tone", "Emboss", "Sepia", "Face Blur"]
FILTER_FUNCS = [defaultGreyscale, blur5x5, coolFilter, embossFilter, filterSepia, faceBlurFilter]

NUM_SEGMENTS = len(FILTER_NAMES)
SEGMENT_ANGLE = 180 / NUM_SEGMENTS
OUTER_RADIUS = 200
INNER_RADIUS = 60

PINCH_THRESHOLD = 40
DOUBLE_PINCH_WINDOW = 0.6
DWELL_TIME = 2.0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

"""
def get_angle_segment(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]

    distance = math.hypot(dx, dy)
    if distance < INNER_RADIUS:
        return None

    angle = math.degrees(math.atan2(-dy, dx))
    angle = max(0, min(180, angle))

    segment = int(angle // SEGMENT_ANGLE)
    segment = min(segment, NUM_SEGMENTS - 1)
    return segment
"""
def get_angle_segment(center, point, reference_angle=0):
    dx = point[0] - center[0]
    dy = point[1] - center[1]

    distance = math.hypot(dx, dy)
    if distance < INNER_RADIUS:
        return None

    angle = math.degrees(math.atan2(-dy, dx))
    angle = (angle - reference_angle + 360) % 360

    if angle > 180:
        return None

    segment = int(angle // SEGMENT_ANGLE)
    segment = min(segment, NUM_SEGMENTS - 1)
    return segment

def draw_radial_menu(frame, center, cursor, active_segment, dwell_progress):
    for i in range(NUM_SEGMENTS):
        start_angle =  - (i + 1) * SEGMENT_ANGLE 
        end_angle =  - i * SEGMENT_ANGLE 

        color = (0, 200, 255) if i == active_segment else (80, 80, 80)

        cv2.ellipse(frame, center, (OUTER_RADIUS, OUTER_RADIUS), 0, start_angle, end_angle, color, -1)

    cv2.circle(frame, center, INNER_RADIUS, (30, 30, 30), -1)

    for i, name in enumerate(FILTER_NAMES):
        mid_angle = math.radians(i * SEGMENT_ANGLE + SEGMENT_ANGLE / 2)
        label_radius = (OUTER_RADIUS + INNER_RADIUS) // 2
        lx = int(center[0] + label_radius * math.cos(mid_angle))
        ly = int(center[1] - label_radius * math.sin(mid_angle))

        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        lx -= text_size[0] // 2
        ly += text_size[1] // 2

        cv2.putText(frame, name, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if active_segment is not None:
        cv2.circle(frame, center, INNER_RADIUS - 10, (0, 255, 0), max(2, int(dwell_progress * 8)))

    cv2.circle(frame, cursor, 8, (0, 255, 0), -1)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mp_draw = mp.solutions.drawing_utils

    menu_open = False
    was_pinching = False
    pinch_timestamps = []

    selected_segment = None
    segment_start_time = None
    active_filter_idx = None

    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        output = frame.copy()
        if active_filter_idx is not None:
            try:
                output = FILTER_FUNCS[active_filter_idx](output)
                if len(output.shape) == 2:
                    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark

            mp_draw.draw_landmarks(output, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]

            center = (
                int((wrist.x + middle_mcp.x) / 2 * WIDTH),
                int((wrist.y + middle_mcp.y) / 2 * HEIGHT),
            )
            thumb_px = (int(thumb_tip.x * WIDTH), int(thumb_tip.y * HEIGHT))
            index_px = (int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT))
            middle_px = (int(middle_tip.x * WIDTH), int(middle_tip.y * HEIGHT))

            pinch_dist = math.hypot(thumb_px[0] - index_px[0], thumb_px[1] - index_px[1])
            is_pinching = pinch_dist < PINCH_THRESHOLD

            if is_pinching and not was_pinching:
                now = time.time()
                pinch_timestamps.append(now)
                pinch_timestamps = [t for t in pinch_timestamps if now - t <= DOUBLE_PINCH_WINDOW]

                if len(pinch_timestamps) >= 2:
                    menu_open = not menu_open
                    pinch_timestamps = []
                    selected_segment = None
                    segment_start_time = None

            was_pinching = is_pinching

            if menu_open:
                segment = get_angle_segment(center, middle_px)

                if segment == selected_segment and segment is not None:
                    elapsed = time.time() - segment_start_time
                    if elapsed >= DWELL_TIME:
                        active_filter_idx = segment
                        menu_open = False
                        selected_segment = None
                        segment_start_time = None
                else:
                    selected_segment = segment
                    segment_start_time = time.time()

                dwell_progress = 0
                if selected_segment is not None and segment_start_time is not None:
                    dwell_progress = min(1.0, (time.time() - segment_start_time) / DWELL_TIME)

                draw_radial_menu(output, center, middle_px, selected_segment, dwell_progress)

            cv2.putText(output, f"Pinch dist: {int(pinch_dist)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output, f"Pinching: {is_pinching}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(output, f"Menu open: {menu_open}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(output, "No hand detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(output, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Filter Menu", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
