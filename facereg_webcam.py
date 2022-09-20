import json
import cv2
import warnings
import pickle
import numpy as np
import face_recognition

from utils.calculate_confidence_score import face_distance_to_conf

warnings.filterwarnings('ignore')


def reg_cam(device=0, model_type="hog", encoding_file=""):
    # Load the known faces and embeddings
    print("[INFO] Loading encodings...")
    try:
        data = pickle.loads(open(encoding_file, "rb").read())
    except Exception as e:
        print("Facial Embeddings file 'encodings.pickle' may not exist.")
        exit()

    # Get a reference to webcam #0 (the default one)
    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print('Cannot open camera!')
        exit()
        
    face_names = []
    face_scores = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1, 1)  # Flip to act as a mirror

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb = small[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(
                img=rgb,
                model=model_type
            )
            face_encodings = face_recognition.face_encodings(
                face_image=rgb,
                known_face_locations=face_locations
            )


            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings=data["encodings"],
                    face_encoding_to_check=face_encoding
                )
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    data["encodings"],
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = data["names"][best_match_index]
                score = face_distance_to_conf(np.min(face_distances))

                face_names.append(name)

                score = np.round(score * 100, 1)
                face_scores.append(score)
                
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name, score in zip(face_locations, face_names, face_scores):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name}", (left + 6, bottom - 8),
                        font, 0.7, (255, 255, 255), 1)
            if name == "Unknown":
                # score = 0
                cv2.putText(frame, f"{score}%", (left + 6, top - 20),
                            font, 0.7, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"{score}%", (left + 6, top - 20),
                            font, 0.7, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF

        # Hit 'q' on the keyboard to quit!
        if key == ord('q') or key == 27:
            break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Streaming stopped...")

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

# py .\facereg_webcam.py
