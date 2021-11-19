import os

import imageio
import numpy as np

from cloud_lib_v2.image_processing import resize4x


class Mask:
    def __init__(self, file_path, resize):
        self.resize = resize
        self.file_path = file_path
        self.mask = self.load_mask()
        self.camera_id = int(os.path.basename(file_path)[7: -4])

    def load_mask(self):
        result = imageio.imread(self.file_path)[:, :, 3][..., np.newaxis]
        if self.resize:
            result = resize4x(result)
        result = (result >= 0.5)
        return result
