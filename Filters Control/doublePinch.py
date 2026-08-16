import cv2
import mediapipe as mp
import numpy as np
import time
import math

from filters import (
    blur5x5, defaultGreyscale, greyscale, filterSepia, filterNegative,
    coolFilter, warmFilter, colorFilter)

WIDTH, HEIGHT = 960, 720
PINCH_THRESHOLD = 30
CONSECUTIVE_FRAMES_REQUIRED = 3 # No of frames to confirm a pinch gesture or change

FILTERS_FUNC = [defaultGreyscale, blur5x5, coolFilter, filterSepia, warmFilter, colorFilter]
FILTERS_NAME = ["Default Grey", "Blur 5x5", "Cool Tone", "Sepia", "Warm Tone", "Color Filter"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

def get_pinch_point(landmarks, image_width, image_height):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    x1, y1 = (thumb_tip.x * image_width), thumb_tip.y * image_height
    x2, y2 = (index_tip.x * image_width), index_tip.y * image_height

    mid = ((x1 + x2) / 2, (y1 + y2) / 2)
    distance = math.hypot(x2 - x1, y2 - y1)

    return mid, distance

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoothed_both_pinch = False
    raw_state_streak = 0
    last_raw_state = False

    filter_index = -1
    box = None

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

        raw_both_pinch = False
        pinch_point = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                mid, distance = get_pinch_point(landmarks, WIDTH, HEIGHT)

                is_pinching = distance < PINCH_THRESHOLD
                if is_pinching:
                    pinch_point.append(mid)

                mid_center = (int(mid[0]), int(mid[1]))
                color = (0, 255, 0) if is_pinching else (0, 0, 255)
                cv2.circle(output, mid_center, 8, color, -1)

            raw_both_pinch = len(pinch_point) == 2

        # Update the smoothed state based on the raw state and the streak count
        if raw_both_pinch == last_raw_state:
            raw_state_streak += 1
        else:
            raw_state_streak = 1
            last_raw_state = raw_both_pinch

        if raw_state_streak >= CONSECUTIVE_FRAMES_REQUIRED and raw_both_pinch != smoothed_both_pinch:
            was_smoothed = smoothed_both_pinch
            smoothed_both_pinch = raw_both_pinch

            if smoothed_both_pinch and not was_smoothed:
                filter_index = (filter_index + 1) % len(FILTERS_FUNC)

        # Resize box if actively smoothed both pinch
        if smoothed_both_pinch and len(pinch_point) == 2:
            xs = [int(p[0]) for p in pinch_point]
            ys = [int(p[1]) for p in pinch_point]

            box = (min(xs), min(ys), max(xs), max(ys))

        if box is not None and filter_index >= 0:
            x_min, y_min, x_max, y_max = box
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(WIDTH, x_max), min(HEIGHT, y_max)

            if x_min < x_max and y_min < y_max:
                roi = np.ascontiguousarray(output[y_min:y_max, x_min:x_max])
                try:
                    filtered = FILTERS_FUNC[filter_index](roi)
                    if len(filtered.shape) == 2:
                        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
                    output[y_min:y_max, x_min:x_max] = filtered
                except Exception:
                    import traceback
                    traceback.print_exc()
            
            cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(output, FILTERS_NAME[filter_index], (x_min + 5, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(output, f"Smoothed Both Pinch: {smoothed_both_pinch}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output, f"Raw Both Pinch: {raw_both_pinch}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output, f"Filter idx: {filter_index}, Box: {box}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        cv2.putText(output, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Double Pinch Filter Control", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()