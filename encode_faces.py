import argparse
import os
import pickle

import cv2
import face_recognition


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to dataset of known faces")
ap.add_argument("-d", "--detection-method", default="hog",
                type=str, help="detection method (hog/cnn")
args = vars(ap.parse_args())

KNOWN_FACES_DIR = args["dataset"]

print("[INFO] locating images...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for count, filename in enumerate(os.listdir(KNOWN_FACES_DIR + os.path.sep + name)):
        print(f"[INFO] processing image {count + 1} of {name}")
        print(filename)

        image = face_recognition.load_image_file(
            KNOWN_FACES_DIR + os.path.sep + name + os.path.sep + filename)
        box = face_recognition.face_locations(
            image, model=args["detection-method"])
        encoding = face_recognition.face_encodings(image, box)
        if len(encoding) != 0:
            known_faces.append(encoding[0])
            known_names.append(name)

print("[INFO] serialising images...")
data = {"encodings": known_faces, "names": known_names}
pickle.dump(data, open("encodings.p", "wb"))
