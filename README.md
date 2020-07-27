# Face-recognition

Face_recognition project that processes a bunch of known images and compares them with unknown images and finds a name for each face found.

To use, first run encode_faces.py passing known faces directory as a CLI argument. 

PLEASE NOTE: directory of known faces must be of format (or similar): known_faces_dir/name1/image.imgextension. Must contain directories of images corresponding to directory's name, corresponding to name of person in images. Each image in known faces should be of 1 person. If not, first face found will be encoded only.

Then, run recognise_faces.py, passing in directory of unknown faces as a CLI arguments. When image is displayed, press any key to move to the next image.

To test whether an image has a recognisable face, run test_face.py passing in image path as an argument.

To test all images in a directory, do same but with test_faces.py

N.B: run python(3) python_file.py -h for more details on CLI arguments