import cv2
import numpy as np
from scipy import stats

# percentiles
percent = [1] + [i for i in range(5, 96, 5)] + [99]


def resize4x(image: np.ndarray):
    img_compressed = cv2.resize(image, dsize=(image.shape[0]//4, image.shape[1]//4))
    return img_compressed


def to_hsv(img: np.ndarray):
    img = img.astype(np.float32) / 255.0
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    # val = 0.2989*r + 0.5870*g+0.1140*b # (SDTV)
    # val = 0.212 * r + 0.701 * g + 0.087 * b  # (Adobe)
    # val = 0.2126 * r + 0.7152 * g + 0.0722 * b  # (HDTV)
    val = 0.2627 * r + 0.6780 * g + 0.0593 * b  # (UHDTV, HDR)
    val = np.clip(val, 0.0, 1.0)
    rgb_max = img.max(axis=-1)
    rgb_min = img.min(axis=-1)
    c = rgb_max - rgb_min
    csm = c + 1e-8
    hue = ((rgb_max == r) * np.mod(g - b, csm) +
           (rgb_max == g) * ((b - r) / csm + 2) +
           (rgb_max == b) * ((r - g) / csm + 4))
    hue = hue * (c > 0)
    hue = hue * 60
    sat = np.zeros_like(rgb_max)
    sat = sat + (val > 0) * c / val
    sat = np.clip(sat, 0.0, 1.0)
    return hue, sat, val


def calculate_features(image):
    features = np.zeros(0)

    for canal in image:
        features = np.hstack(
            [
                features,
                np.mean(canal),  # mean
                np.var(canal),  # variance
                np.max(canal),  # max
                np.min(canal),  # min
                stats.skew(canal, axis=None),  # excess
                stats.kurtosis(canal, axis=None),  # asymmetry
                np.percentile(canal, percent)  # percentiles
            ]
        )
    return features


def compute_correlation(a, b):
    return np.corrcoef(a, b)[0, 1]
