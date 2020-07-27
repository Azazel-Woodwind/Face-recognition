import argparse
import face_recognition
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tolerance", default=0.6, type=float,
                help="Tolerance of image recognition")
ap.add_argument("-d", "--detection_method", default="hog",
                type=str, help="Detection method (hog/cnn")
ap.add_argument("-i", "--images", required=True,
                help="path to directory of unknown faces")
args = vars(ap.parse_args())

UNKNOWN_FACES_DIR = args["images"]


print("[INFO] loading encodings...")
data = pickle.load(open("encodings.p", "rb"))
print("[INFO] recognising faces")

for filename in os.listdir(UNKNOWN_FACES_DIR):

    image = face_recognition.load_image_file(
        UNKNOWN_FACES_DIR + os.path.sep + filename)

    boxes = face_recognition.face_locations(
        image, model=args["detection_method"])
    encodings = face_recognition.face_encodings(image, boxes)

    for encoding, locations in zip(encodings, boxes):
        matches = face_recognition.compare_faces(
            data["encodings"], encoding, tolerance=args["tolerance"])
        name = "Unknown"
        names = {}

        if True in matches:
            ids = [i for (i, b) in enumerate(matches) if b]
            for i in ids:
                name = data["names"][i]
                names[name] = names.get(name, 0) + 1

            name = max(names, key=names.get)

        top = locations[0]
        right = locations[1]
        bottom = locations[2]
        left = locations[3]

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow(filename, image)
    cv2.waitKey(0)
