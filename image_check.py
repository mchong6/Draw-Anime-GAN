import cv2
import sys
import os
from colorthief import ColorThief

''' Takes in an image and check if there is a face.
    If not, move it to a temporaru folder to check'''
def detect(filename, file, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        return 0
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    if len(faces) == 0:
        print(filename)
        # move bad images to another folder to manually check
        os.rename(filename, './faces_temp/'+file)
        #os.remove(filename)
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

''' Takes in an image and check if its is black and white.
    Remove those images'''
def manga(filename):
    color_thief = ColorThief(filename)
    x,y,z = color_thief.get_color(quality=1)
    # Naive way to check for black and white
    if abs(x-y)<5 and abs(y-z) < 5 and abs(x-z) < 5:
        global count 
        count += 1
        print count, filename
        os.remove(filename)


count = 0
imdir = "./faces"
for root, dirs, files in os.walk(imdir):
    path = root.split(os.sep)
    for file in files:
        filePath = os.path.join(os.path.abspath(root), file)
        detect(filePath, file)
        #manga(filePath)
