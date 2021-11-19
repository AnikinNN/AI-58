import numpy as np


def resize4x(img):
    img = img.astype(np.float32)
    img_compressed = img[::4, :, :] + img[1::4, :, :] + img[2::4, :, :] + img[3::4, :, :]
    img_compressed = img_compressed[:, ::4, :] + img_compressed[:, 1::4, :] + img_compressed[:, 2::4,
                                                                              :] + img_compressed[:, 3::4, :]
    img_compressed = img_compressed // 16
    return img_compressed
