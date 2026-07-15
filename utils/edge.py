import cv2
import numpy as np

def edge_statistics(image):

    img = (image*255).astype(np.uint8)

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)

    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

    grad = np.sqrt(sobelx**2 + sobely**2)

    return {

        "Edge Strength":
            round(np.mean(grad),2),

        "Max Gradient":
            round(np.max(grad),2)
    },grad