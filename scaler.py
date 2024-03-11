import cv2 as cv

def scaleDownImage(image, scale):
    return cv.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))