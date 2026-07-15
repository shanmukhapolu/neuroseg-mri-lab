import cv2
import numpy as np

def overlay(image, mask, alpha=0.45):

    rgb = cv2.cvtColor(
        (image*255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )

    color = np.zeros_like(rgb)

    color[:,:,0] = 255

    mask_bool = mask>0

    rgb[mask_bool] = cv2.addWeighted(
        rgb[mask_bool],
        1-alpha,
        color[mask_bool],
        alpha,
        0
    )

    return rgb