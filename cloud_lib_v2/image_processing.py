import numpy as np
from scipy import stats

# percentiles
percent = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]


def resize4x(img):
    img = img.astype(np.float32)
    img_compressed = img[::4, :, :] + img[1::4, :, :] + img[2::4, :, :] + img[3::4, :, :]
    img_compressed = img_compressed[:, ::4, :] + img_compressed[:, 1::4, :] + img_compressed[:, 2::4,
                                                                              :] + img_compressed[:, 3::4, :]
    img_compressed = img_compressed // 16
    return img_compressed


def HSV(img: np.ndarray):
    img = img.astype(np.float32) / 255.0
    r = img[:, 0]
    g = img[:, 1]
    b = img[:, 2]
    # val = 0.2989*r + 0.5870*g+0.1140*b # (SDTV)
    # val = 0.212 * r + 0.701 * g + 0.087 * b  # (Adobe)
    # val = 0.2126 * r + 0.7152 * g + 0.0722 * b  # (HDTV)
    val = 0.2627 * r + 0.6780 * g + 0.0593 * b  # (UHDTV, HDR)
    val = np.clip(val, 0.0, 1.0)
    rgbmax = img.max(axis=-1)
    rgbmin = img.min(axis=-1)
    C = rgbmax - rgbmin
    hue = np.zeros_like(rgbmax)
    Csm = C + 1e-8
    hue = ((rgbmax == r) * np.mod(g - b, Csm) +
           (rgbmax == g) * ((b - r) / Csm + 2) +
           (rgbmax == b) * ((r - g) / Csm + 4))
    hue = hue * (C > 0)
    hue = hue * 60
    sat = np.zeros_like(rgbmax)
    sat = sat + (val > 0) * C / val
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
                stats.skew(canal),  # excess
                stats.kurtosis(canal),  # asymmetry
                np.percentile(canal, percent)  # percentiles
            ]
        )
    return features


def compute_correlation(a, b):
    return np.corrcoef(a, b)[0, 1]
