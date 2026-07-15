import cv2
import numpy as np


def sobel_gradient(image):

    img = (image * 255).astype(np.uint8)

    gx = cv2.Sobel(
        img,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    grad = np.sqrt(gx**2 + gy**2)

    grad /= grad.max() + 1e-8

    return grad


def laplacian(image):

    img = (image * 255).astype(np.uint8)

    lap = cv2.Laplacian(
        img,
        cv2.CV_64F
    )

    lap = np.abs(lap)

    lap /= lap.max() + 1e-8

    return lap


def canny(image):

    img = (image * 255).astype(np.uint8)

    edges = cv2.Canny(
        img,
        50,
        150
    )

    return edges / 255.0