import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy import ma


# для расчета персентилей
percent = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]


def get_masked_image(photo_path, masks):
    mask = np.tile(masks[int(os.path.abspath(photo_path)[-5]) - 1], (1, 1, 3))
    img = imageio.imread(photo_path)
    img = resize4x(img).astype(np.uint8)

    img_masked = np.reshape(ma.array(img, mask=mask, ),
                            (-1, 3)
                            )

    canals = list(img_masked[:, j] for j in range(3))
    # add hsv canals
    canals.extend(HSV(img_masked))

    return canals


class features_calculator:
    def __init__(self, masks):
        self.masks = masks

    def __call__(self, x, *args, **kwargs):
        return calculate_masked_features(x, self.masks)


def resize4x(img):
    img = img.astype(np.float32)
    img_compressed = img[::4, :, :] + img[1::4, :, :] + img[2::4, :, :] + img[3::4, :, :]
    img_compressed = img_compressed[:, ::4, :] + img_compressed[:, 1::4, :] + img_compressed[:, 2::4,
                                                                              :] + img_compressed[:, 3::4, :]
    img_compressed = img_compressed // 16
    return img_compressed


def load_masks(mask_paths):
    masks = []
    for mask in mask_paths:
        m1 = imageio.imread(mask)[:, :, 3][..., np.newaxis]
        m1 = resize4x(m1)
        m1 = (m1 >= 0.5)  # .astype(np.uint8)
        # m1 = np.tile(m1, (1,1,3))
        masks.append(m1)
    return masks


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


def calculate_features(image: ma.array):
    features = np.zeros(0)
    canals = list(image[:, j] for j in range(3))
    # Добавим hsv каналы
    canals.extend(HSV(image))

    for canal in canals:
        features = np.hstack(
            [
                features,
                np.mean(canal),  # среднее
                np.var(canal),  # дисперсия
                np.max(canal),  # максимум
                np.min(canal),  # минимум
                stats.skew(canal),  # эксцесс
                stats.kurtosis(canal),  # ассиметрия
                np.percentile(canal, percent)  # персентили
            ]
        )
    return features


def calculate_masked_features(photo_path, masks):
    mask = np.tile(masks[int(os.path.abspath(photo_path)[-5]) - 1], (1, 1, 3))
    img = imageio.imread(photo_path)
    img = resize4x(img).astype(np.uint8)

    img_masked = np.reshape(ma.array(img,
                                     mask=mask,
                                     ),
                            (-1, 3)
                            )
    return calculate_features(img_masked)


if __name__ == '__main__':
    from path import *
    path_to_photo = os.path.join(photos_base_dir, r'snapshots-2021-08-03\img-2021-08-03T00-00-33devID2.jpg')
    # возвращает массив из 108 признаков
    calculated_features = calculate_masked_features(path_to_photo, masks)

    mask = np.tile(masks[0], (1, 1, 3))
    white_img = np.reshape(ma.array(np.ones([480, 480, 3]) * 255,
                                    mask=mask,
                                    ),
                           (-1, 3)
                           )
    white_features = calculate_features(white_img)
    print(white_features)

    # test for RGB canals
    for i in range(0, len(white_features) // 2, len(white_features) // 6):
        assert white_features[i] == 255.0
        assert white_features[i + 1] == 0.0
        assert white_features[i + 2] == 255.0
        assert white_features[i + 3] == 255.0
        # чему равны эксцесс и ассиметрия в таком случае???
        # assert white_features[i + 4] == 255.0
        # assert white_features[i + 5] == 255.0
        for p in range(len(percent)):
            assert white_features[i + 6 + p] == 255.0
