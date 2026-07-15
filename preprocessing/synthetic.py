import cv2
import numpy as np
from skimage.filters import gaussian

def generate_synthetic_mri():

    img = np.zeros((512,512),dtype=np.float32)

    cv2.ellipse(
        img,
        (256,256),
        (180,220),
        0,
        0,
        360,
        0.45,
        -1
    )

    cv2.circle(
        img,
        (340,180),
        40,
        0.9,
        -1
    )

    cv2.circle(
        img,
        (320,170),
        25,
        0.85,
        -1
    )

    img = gaussian(img,sigma=2)

    noise = np.random.normal(0,0.02,img.shape)

    img = np.clip(img+noise,0,1)

    return img