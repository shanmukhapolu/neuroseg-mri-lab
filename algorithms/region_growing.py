import numpy as np
from collections import deque

def segment(image, seed, tolerance):

    h, w = image.shape

    mask = np.zeros((h, w), dtype=np.uint8)

    sy, sx = seed

    if sy < 0 or sy >= h or sx < 0 or sx >= w:
        return mask

    seed_value = image[sy, sx]

    q = deque()
    q.append((sy, sx))

    mask[sy, sx] = 255

    neighbors = [
        (-1,0),
        (1,0),
        (0,-1),
        (0,1)
    ]

    while q:

        y,x = q.popleft()

        for dy,dx in neighbors:

            ny = y+dy
            nx = x+dx

            if 0 <= ny < h and 0 <= nx < w:

                if mask[ny,nx] == 0:

                    if abs(image[ny,nx]-seed_value) <= tolerance:

                        mask[ny,nx] = 255
                        q.append((ny,nx))

    return mask