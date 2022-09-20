import cv2


def draw_rect(frame, face, length=30, thickness=5):
    x, y, w, h = face
    x1, y1 = x + w,  y + h

    cv2.rectangle(frame, face, (0, 255, 0), 1)

    # Top left: x, y
    cv2.line(frame, (x, y), (x + length, y), (0, 255, 0), thickness)
    cv2.line(frame, (x, y), (x, y + length), (0, 255, 0), thickness)

    # Top right: x1, y
    cv2.line(frame, (x1, y), (x1 - length, y), (0, 255, 0), thickness)
    cv2.line(frame, (x1, y), (x1, y + length), (0, 255, 0), thickness)

    # Bottom left: x, y1
    cv2.line(frame, (x, y1), (x + length, y1), (0, 255, 0), thickness)
    cv2.line(frame, (x, y1), (x, y1 - length), (0, 255, 0), thickness)

    # Bottom right: x1, y1
    cv2.line(frame, (x1, y1), (x1 - length, y1), (0, 255, 0), thickness)
    cv2.line(frame, (x1, y1), (x1, y1 - length), (0, 255, 0), thickness)

    return frame
