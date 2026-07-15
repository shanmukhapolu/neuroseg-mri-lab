from skimage.feature import graycomatrix, graycoprops
import numpy as np

def texture_features(image, mask):

    region = image[mask > 0]

    if len(region) < 10:
        return {}

    roi = (image * 255).astype(np.uint8)

    glcm = graycomatrix(
        roi,
        [1],
        [0],
        256,
        symmetric=True,
        normed=True
    )

    return {

        "Contrast":
            round(graycoprops(glcm,"contrast")[0,0],3),

        "Correlation":
            round(graycoprops(glcm,"correlation")[0,0],3),

        "Energy":
            round(graycoprops(glcm,"energy")[0,0],3),

        "Homogeneity":
            round(graycoprops(glcm,"homogeneity")[0,0],3)
    }