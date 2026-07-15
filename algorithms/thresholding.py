import cv2
import numpy as np

def otsu(image):

    img = (image*255).astype(np.uint8)

    _,mask = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    return mask