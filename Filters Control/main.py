import cv2
import numpy as np
from filters import (
    blur5x5, defaultGreyscale, greyscale, filterSepia, filterNegative,
    coolFilter, warmFilter, colorFilter,
    blurFilter, sobelX, sobelY, magnitudeSobel, embossFilter, faceFinder
)

FILTERS = {
    ord('1'): ("Default Greyscale", defaultGreyscale),
    ord('2'): ("Custom Greyscale", greyscale),
    ord('3'): ("Sepia", filterSepia),
    ord('4'): ("Negative", filterNegative),
    ord('5'): ("Cool Tone", coolFilter),
    ord('6'): ("Warm Tone", warmFilter),
    ord('7'): ("Blur ", blurFilter),
    ord('8'): ("Blur 5x5", blur5x5),
    ord('9'): ("Emboss", embossFilter),
    ord('0'): ("Sobel Magnitude", magnitudeSobel),
    ord('f'): ("Face Finder", faceFinder),
}

WIDTH, HEIGHT = 960, 720


def draw_legend(frame):
    y = HEIGHT - 20 * (len(FILTERS) + 1)
    cv2.putText(frame, "0-9,f: filters  n: none  q: quit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20
    for key, (name, _) in FILTERS.items():
        cv2.putText(frame, f"{chr(key)}: {name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    current_key = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        if current_key == ord('0'):
            sx = sobelX(frame)
            sy = sobelY(frame)
            output = magnitudeSobel(sx, sy)
        elif current_key in FILTERS and FILTERS[current_key][1] is not None:
            output = FILTERS[current_key][1](frame)
        else:
            output = frame

        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        draw_legend(output)
        cv2.imshow("Filters Debug", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_key = None
        elif key in FILTERS:
            current_key = key

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()