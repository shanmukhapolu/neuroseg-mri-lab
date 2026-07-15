import numpy as np
from scipy.stats import entropy

def intensity_statistics(image, mask):

    pixels = image[mask>0]

    if len(pixels)==0:
        return {}

    hist,_ = np.histogram(
        pixels,
        bins=32,
        range=(0,1),
        density=True
    )

    return {

        "Mean Intensity":
            round(np.mean(pixels),3),

        "Median":
            round(np.median(pixels),3),

        "Std Dev":
            round(np.std(pixels),3),

        "Minimum":
            round(np.min(pixels),3),

        "Maximum":
            round(np.max(pixels),3),

        "Entropy":
            round(entropy(hist+1e-8),3)

    }