import cv2
import sys
import os

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    if len(faces) == 0:
        print(filename)
        os.remove(filename)
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite("out.png", image)

imdir = "./danbooru-faces"
for root, dirs, files in os.walk(imdir):
    path = root.split(os.sep)
    for file in files:
        filePath = os.path.join(os.path.abspath(root), file)
        detect(filePath)
