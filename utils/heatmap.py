
import cv2
import numpy as np


def turbo(image):

    img = (image * 255).astype(np.uint8)

    return cv2.cvtColor(
        cv2.applyColorMap(
            img,
            cv2.COLORMAP_TURBO
        ),
        cv2.COLOR_BGR2RGB
    )


def inferno(image):

    img = (image * 255).astype(np.uint8)

    return cv2.cvtColor(
        cv2.applyColorMap(
            img,
            cv2.COLORMAP_INFERNO
        ),
        cv2.COLOR_BGR2RGB
    )


def magma(image):

    img = (image * 255).astype(np.uint8)

    return cv2.cvtColor(
        cv2.applyColorMap(
            img,
            cv2.COLORMAP_MAGMA
        ),
        cv2.COLOR_BGR2RGB
    )