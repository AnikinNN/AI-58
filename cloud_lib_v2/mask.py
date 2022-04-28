import os
import cv2
import numpy as np

from cloud_lib_v2.image_processing import resize4x


class Mask:
    def __init__(self, file_path, resize):
        self.resize = resize
        self.file_path = file_path
        self.mask = self.load_mask()
        self.camera_id = int(os.path.basename(file_path)[7: -4])

    def load_mask(self):
        image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        if self.resize:
            image = resize4x(image)
        flat_mask = np.where(image > 0, True, False)
        return np.repeat(flat_mask[:, :, np.newaxis], 3, axis=2)
