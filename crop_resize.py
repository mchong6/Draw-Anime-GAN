from __future__ import print_function
from multiprocessing import Pool
from PIL import Image
import numpy as np
import sys
import os
import cv2


# im from PIL.Image.open, face_pos position object, margin
def faceCrop(im,face_pos,m):
    """
    m is the relative margin added to the face image
    """
    if not im:
        return 0
    x,y,w,h = face_pos
    sizeX, sizeY = im.size
    new_x, new_y = max(0,x-m*w), max(0,y-m*h)
    new_w = w + 2*m*w if sizeX > (new_x + w + 2*m*w) else sizeX - new_x
    new_h = h + 2*m*h if sizeY > (new_y + h + 2*m*h) else sizeY - new_y
    new_x,new_y,new_w,new_h = int(new_x),int(new_y),int(new_w),int(new_h)
    return im.crop((new_x,new_y,new_x+new_w,new_y+new_h))
    
def min_resize_crop(im, min_side):
    sizeX,sizeY = im.size
    if not sizeX or not sizeY:
        return None
    if sizeX > sizeY:
        im = im.resize((min_side*sizeX/sizeY, min_side), Image.ANTIALIAS)
    else:
        im = im.resize((min_side, sizeY*min_side/sizeX), Image.ANTIALIAS)
    return im.crop((0,0,min_side,min_side))
    #return im

def load_detect(img_path, new_img_path, cascade_file = "./lbpcascade_animeface.xml"):
    """Read original image file, return the cropped face image in the size 96x96

    Input: A string indicates the image path
    Output: Detected face image in the size 96x96

    Note that there might be multiple faces in one image, 
    the output crossponding to the face with highest probability
    """
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(img_path)
    im = Image.open(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (48, 48))
    if not len(faces):
        return 0
    for x,y,w,h in faces:
        #1.5x scaling of crop
        w_add = .125 * w
        h_add = .125 * h
        w = w - 2*w_add
        h = h - 2*h_add
        x += w_add 
        y += h_add
        x,y,w,h = int(x), int(y), int(w), int(h)
        face_pos = [x,y,w,h]
        im = faceCrop(im, face_pos, 0.5)
        if not im:
            return 0
        im = min_resize_crop(im, outputSize)
        if im:
            im.save(new_img_path, 'JPEG')

def process_img(img_path):
    """
    The face images are stored in {${pwd} + faces} 
    """
    tmp = img_path.split('/')
    img_name = tmp[len(tmp)-1]
    new_dir_path = './faces'

    new_img_path = os.path.join(new_dir_path, img_name)
    if os.path.exists(new_img_path):
        return 0
    im = load_detect(img_path, new_img_path)

def try_process_img(img_path):
    process_img(img_path)

# multiprocessing version
def multi_construct_face_dataset(base_dir):
    imgs = [os.path.join(base_dir,f) for f in os.listdir(base_dir) if f.endswith(('.jpg', '.png'))]
    print('There are %d images in total. \n' % (len(imgs)))
    for i in imgs:
        try_process_img(i)


base_dir = './dataset'
outputSize = 128
multi_construct_face_dataset(base_dir)
