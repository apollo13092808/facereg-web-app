import os
import cv2
import warnings
import argparse
import mediapipe as mp
from utils.change_resolution import make_480p

warnings.filterwarnings('ignore')

class FaceDetector:
    def __init__(self, min_conf=0.75) -> None:
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_conf
        )

    def detect_faces(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(rgb)

        bboxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                f_height, f_width, f_channels = frame.shape
                bbox = (
                    int(bboxC.xmin * f_width),
                    int(bboxC.ymin * f_height),
                    int(bboxC.width * f_width),
                    int(bboxC.height * f_height)
                )
                bboxes.append([id, bbox, detection.score])

                if draw:
                    frame = self.draw_rectangle(frame, bbox)

                    # cv2.putText(
                    #     img=frame,
                    #     text=f"{int(detection.score[0] * 100)}%",
                    #     org=(bbox[0], bbox[0] - 15),
                    #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    #     fontScale=1,
                    #     color=(255, 255, 255),
                    #     thickness=2
                    # )

        return frame, bboxes

    def draw_rectangle(self, frame, bbox, length=30, thickness=5):
        x, y, w, h = bbox
        x1, y1 = x + w,  y + h

        cv2.rectangle(frame, bbox, (0, 255, 0), 1)

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


def main(device=0):
    print("[INFO] Initializing face capture. Look at the camera/webcam and wait...")

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("Camera/webcam cannot be opened or video file corrupted.")
        exit()

    make_480p(cap)

    detector = FaceDetector()

    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.flip(frame, 1, 1)  # Flip to act as a mirror

            frame, bboxes = detector.detect_faces(frame)

            cv2.imshow("Frame", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                print("[INFO] Face Capturing Finished...")
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
