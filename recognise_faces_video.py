from imutils.video import VideoStream
import face_recognition
import pickle
import argparse
import time
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
                help="0 to prevent display of frames to screen")
ap.add_argument("-d", "--detection", default="hog",
                type=str, help="Detection method (hog/cnn")
args = vars(ap.parse_args())


print("[INFO] loading encodings...")

data = pickle.load(open("encodings.p", "rb"))

print("[INFO] starting video stream...")
vs = VideoStream().start()
writer = None
time.sleep(3)

while True:

    frame = vs.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model=args["detection"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, locations in zip(encodings, boxes):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        names = {}

        if True in matches:
            ids = [i for (i, b) in enumerate(matches) if b]
            for i in ids:
                name = data["names"][i]
                names[name] = names.get(name, 0) + 1

            name = max(names, key=names.get)

        for (top, right, bottom, left) in boxes:

            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, 20, (frame.shape[1], frame.shape[2]), True)

    if writer is not None:
        writer.write(frame)

    if args["display"] == 1:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

cv2.destroyAllWindows()
vs.stop()

if writer is not None:
    writer.release()
