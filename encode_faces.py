import os
import cv2
import pickle
import argparse
import warnings
import face_recognition
from imutils import paths
import time

start = time.time()

warnings.filterwarnings('ignore')


def encode_face(dataset_path="datasets", model_type="cnn", encoding_file="encodings.pickle"):
    # Grab the paths to the input images in our dataset
    print("[INFO] Quantifying faces...")
    image_paths = list(paths.list_images(dataset_path))  # args["input"]

    # Initialize the list of known encodings and known names
    known_encodings = []
    known_names = []

    # Loop over the image paths
    for (i, img_path) in enumerate(image_paths):
        # Extract the person name from the image path
        print(f"[INFO] Processing image {i + 1}/{len(image_paths)}")
        name = img_path.split(os.path.sep)[-2][:-5]

        # Load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the (x, y) coordinates of the bounding boxes corresponding to each face in the input image
        face_locations = face_recognition.face_locations(
            img=rgb,
            model=model_type  # args["model"]
        )

        # Compute the facial embeddings for the face
        encodings = face_recognition.face_encodings(rgb, face_locations)

        # Loop over the encodings
        for encoding in encodings:
            # Add each encoding + name to our set of known names and encodings
            known_encodings.append(encoding)
            known_names.append(name)

    # Dump the facial encodings + names to disk
    print("[INFO] Serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    f = open(encoding_file, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("[INFO] Done !!!")

    seconds = time.time() - start
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))


if __name__ == '__main__':
    encode_face()

# py .\encode_faces.py
