import json
import cv2
import pickle
import warnings
import face_recognition
import numpy as np

from utils.calculate_confidence_score import face_distance_to_conf

warnings.filterwarnings('ignore')


def reg_image(img_path="", model_type="hog", encoding_file=""):
    # Load the known faces and embeddings
    print("[INFO] Loading encodings...")
    try:
        data = pickle.loads(open(encoding_file, "rb").read())
    except Exception as e:
        print("Facial Embeddings file 'encodings.pickle' may not exist.")

    # Load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y) coordinates of the bounding boxes corresponding to each face in the input image
    print("[INFO] Recognizing faces...")
    boxes = face_recognition.face_locations(
        img=rgb,
        model=model_type
    )

    # Compute the facial embeddings for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Initialize the list of names for each face detected
    names = []
    scores = []

    # Loop over the encodings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # # Check to see if we have found a match
        # if True in matches:
        #     # Find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
        #     matched_indices = [i for (i, b) in enumerate(matches) if b]
        #     counts = {}

        #     # Loop over the matched indexes and maintain a count for each recognized face face
        #     for i in matched_indices:
        #         name = data["names"][i]
        #         counts[name] = counts.get(name, 0) + 1

        #     # Determine the recognized face with the largest number of votes (note: in the event of an unlikely tie Python will
        #     # select first entry in the dictionary)
        #     name = max(counts, key=counts.get)

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            data["encodings"],
            encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = data["names"][best_match_index]
        
        score = face_distance_to_conf(np.min(face_distances))

        # update the list of names
        names.append(name)
        score = np.round(score*100, 1)
        scores.append(score)

    # Loop over the recognized faces
    for ((top, right, bottom, left), name, score) in zip(boxes, names, scores):
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{name}", (left + 6, bottom - 8),
                    font, 0.7, (255, 255, 255), 1)
        if name == "Unknown":
            cv2.putText(image, f"{0}%", (left + 6, top - 20),
                        font, 0.7, (255, 255, 255), 1)
        else:
            cv2.putText(image, f"{score}%", (left + 6, top - 20),
                        font, 0.7, (255, 255, 255), 1)

    # Show the output image

    def resize_image_with_aspect_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = img.shape[:2]

        if width is None and height is None:
            return img

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(src=img, dsize=dim, interpolation=inter)

    # Data to processed
    json_dict = {
        "name": name,
        "score": score,
    }
    if name == "Unknown":
        pass
    else:
        # Serializing json
        json_obj = json.dumps(json_dict, indent=4)

        # Writing to sample.json
        with open("sample.json", "w") as json_file:
            json_file.write(json_obj)

    print("[INFO] Finished...")
    resized = resize_image_with_aspect_ratio(img=image, width=600)
    cv2.imshow("Image", resized)
    # cv2.imshow("Image", image)
    cv2.waitKey()

# py .\facereg_image.py -i .\test\cv.jpg -e .\encodings.pickle -m hog


if __name__ == "__main__":
    reg_image(img_path="", model_type="hog", encoding_file="")
