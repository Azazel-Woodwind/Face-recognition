import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required = False, help = "your name")
args = vars(ap.parse_args())

f = open("testFIle.txt", "r")
print (f.read())
print (f.readlines())
