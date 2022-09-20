import os
import cv2
import warnings
from utils.face_detector_mediapipe import FaceDetector
from utils.change_resolution import make_480p
from utils.generate_id import generate
from utils.standardize_name import standardize

                                                   
warnings.filterwarnings('ignore')


def generate_dataset(device=0, c=20):
    # For each person, enter one their name
    face_name = input("* Enter user's name: ")
    face_name = standardize(face_name)

    # Unique ID for each person
    unique_id = str(generate())

    # Create dataset folder for each person
    dataset_path = os.path.join("datasets", f"{face_name} {unique_id}")
    os.mkdir(dataset_path)

    print("[INFO] Initializing face capture. Look at the camera/webcam and wait...")

    cap = cv2.VideoCapture(device)  # args["device"]

    if not cap.isOpened():
        print("Camera/webcam cannot be opened or video file corrupted.")
        exit()

    # Initialize individual sampling face count
    count = 0

    make_480p(cap)

    detector = FaceDetector()

    print("[INFO] Video is streaming...")
    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.flip(frame, 1, 1)  # Flip to act as a mirror

            frame, bboxes = detector.detect_faces(frame, False)

            cv2.imshow("Frame", frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for bbox in bboxes:
                (x, y, w, h) = bbox[1]

                if count >= c:  # args["count"]:
                    print(f"[INFO] {count} images saved. You can stop streaming now.")
                    break

                count += 1

                file_name = f"{count}.jpg"
                cv2.imwrite(
                    filename=os.path.join(dataset_path, file_name),
                    img=gray[y-60:y+h+50, x-50:x+w+50]  # face area
                )

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                print("[INFO] Face capturing finished...")
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    generate_dataset(device=0)

# py .\generate_datasets_mediapipe.py 
