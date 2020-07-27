import os
import face_recognition
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to directory of images you want to test")
args = vars(ap.parse_args())

IMAGES_DIR = args["images"]

bad_images = []

for count, filename in enumerate(os.listdir(IMAGES_DIR)):
    print(f"Checking image {count + 1}/{len(os.listdir(IMAGES_DIR))}")

    image = cv2.imread(IMAGES_DIR + os.path.sep + filename)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    box = face_recognition.face_locations(rgb, model="hog")
    encoding = face_recognition.face_encodings(rgb, box)

    if len(encoding) == 0:
        print("No face found")
        bad_images.append(filename)
    else:
        print("Image good")

if len(bad_images) == 0:
    print("All images good")
else:
    print("No face found in images:")
    for image in bad_images:
        print(image)
