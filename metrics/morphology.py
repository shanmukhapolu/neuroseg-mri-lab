import cv2
import numpy as np
from scipy.spatial.distance import pdist


def analyze(mask: np.ndarray) -> dict:
    """
    Compute morphological features from a binary segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255).

    Returns
    -------
    dict
        Dictionary of morphology metrics.
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return {}

    contour = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    x, y, w, h = cv2.boundingRect(contour)

    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = 0
        cy = 0

    # -------------------------------
    # Shape Metrics
    # -------------------------------

    circularity = (
        (4 * np.pi * area) /
        (perimeter ** 2 + 1e-8)
    )

    irregularity = (
        (perimeter ** 2) /
        (4 * np.pi * area + 1e-8)
    )

    solidity = (
        area /
        (hull_area + 1e-8)
    )

    convexity = (
        hull_perimeter /
        (perimeter + 1e-8)
    )

    compactness = (
        perimeter ** 2 /
        (area + 1e-8)
    )

    border_roughness = (
        perimeter /
        (2 * np.sqrt(np.pi * area + 1e-8))
    )

    equivalent_diameter = np.sqrt(
        4 * area / np.pi
    )

    aspect_ratio = (
        w /
        (h + 1e-8)
    )

    extent = (
        area /
        (w * h + 1e-8)
    )

    # -------------------------------
    # Ellipse Features
    # -------------------------------

    major_axis = None
    minor_axis = None
    orientation = None
    eccentricity = None

    if len(contour) >= 5:

        ellipse = cv2.fitEllipse(contour)

        (_, _), (axis1, axis2), angle = ellipse

        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)

        orientation = angle

        if major_axis > 0:
            eccentricity = np.sqrt(
                1 - (minor_axis ** 2) / (major_axis ** 2)
            )

    # -------------------------------
    # Feret Diameter
    # -------------------------------

    feret = None

    points = contour[:, 0, :]

    if len(points) > 1:
        feret = np.max(pdist(points))

    # -------------------------------
    # Enclosing Circle
    # -------------------------------

    (_, _), radius = cv2.minEnclosingCircle(contour)

    enclosing_circle_area = np.pi * radius ** 2

    # -------------------------------
    # Rectangle Metrics
    # -------------------------------

    rect_area = w * h

    rectangularity = (
        area /
        (rect_area + 1e-8)
    )

    # -------------------------------
    # Convex Hull Defects
    # -------------------------------

    hull_indices = cv2.convexHull(
        contour,
        returnPoints=False
    )

    num_defects = 0

    if (
        hull_indices is not None
        and len(hull_indices) > 3
    ):

        defects = cv2.convexityDefects(
            contour,
            hull_indices
        )

        if defects is not None:
            num_defects = defects.shape[0]

    # -------------------------------
    # Final Dictionary
    # -------------------------------

    return {

        "Area (px²)": round(area, 2),

        "Perimeter (px)": round(perimeter, 2),

        "Circularity": round(circularity, 4),

        "Irregularity Index": round(irregularity, 4),

        "Compactness": round(compactness, 4),

        "Solidity": round(solidity, 4),

        "Convexity": round(convexity, 4),

        "Border Roughness": round(border_roughness, 4),

        "Equivalent Diameter": round(
            equivalent_diameter,
            2
        ),

        "Feret Diameter": (
            round(feret, 2)
            if feret is not None
            else "-"
        ),

        "Bounding Box": f"{w} × {h}",

        "Bounding Box Area": rect_area,

        "Aspect Ratio": round(
            aspect_ratio,
            3
        ),

        "Extent": round(
            extent,
            4
        ),

        "Rectangularity": round(
            rectangularity,
            4
        ),

        "Convex Hull Area": round(
            hull_area,
            2
        ),

        "Convex Hull Perimeter": round(
            hull_perimeter,
            2
        ),

        "Convexity Defects": num_defects,

        "Enclosing Circle Radius": round(
            radius,
            2
        ),

        "Enclosing Circle Area": round(
            enclosing_circle_area,
            2
        ),

        "Major Axis": (
            round(major_axis, 2)
            if major_axis is not None
            else "-"
        ),

        "Minor Axis": (
            round(minor_axis, 2)
            if minor_axis is not None
            else "-"
        ),

        "Eccentricity": (
            round(eccentricity, 4)
            if eccentricity is not None
            else "-"
        ),

        "Orientation (°)": (
            round(orientation, 2)
            if orientation is not None
            else "-"
        ),

        "Centroid": (
            round(cx, 2),
            round(cy, 2)
        )

    }