import os
import cv2
import argparse
import face_recognition

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

IMAGE_PATH = args["image"]

image = cv2.imread(IMAGE_PATH)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

box = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, box)

if len(encodings) == 0:
    print("face not found")
else:
    print("face found")
