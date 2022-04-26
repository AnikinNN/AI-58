import os
import threading

import numpy as np
import pandas as pd
import cv2
import torch
from imgaug import SegmentationMapsOnImage, augmenters

from sklearn.utils import shuffle
from torchvision.transforms import transforms

from regressor_on_resnet.threadsafe_iterator import ThreadsafeIterator
# from skimage import io


def get_object_index(objects_count):
    """Cyclic generator of indices from 0 to objects_count
    """
    current_id = 0
    while True:
        yield current_id
        current_id = (current_id + 1) % objects_count


class FluxDataset:
    def __init__(self, flux_frame, batch_size=32, do_shuffle=True, do_augment=True):

        self.flux_frame = flux_frame
        self.mask_dict = {}

        self.do_shuffle = do_shuffle
        self.do_augment = do_augment
        self.batch_size = batch_size

        self.objects_id_generator = ThreadsafeIterator(get_object_index(self.flux_frame.shape[0]))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0

        self.augmentation_sequence = augmenters.Sequential([
            augmenters.Fliplr(0.5),  # 50% of images will be flipped(left-right)
            augmenters.Flipud(0.5),  # -.- (up-down)
            # augmenters.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
            augmenters.Affine(shear=(-16, 16), rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            augmenters.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
        ], random_order=True)

        self.output_size = (512, 512)

    def __len__(self):
        return self.flux_frame.shape[0]

    def shuffle_data(self):
        if self.do_shuffle:
            self.flux_frame = shuffle(self.flux_frame)

    def get_data_by_id(self, index):
        photo_path = self.flux_frame.iloc[index]['photo_path']
        image = self.imread(photo_path)
        mask = self.get_mask(photo_path)
        flux = self.flux_frame.iloc[index]['CM3up[W/m2]']

        sample = {'image': image,
                  'flux': flux,
                  'mask': mask}
        return sample

    def get_mask(self, photo_path):
        # /dasio/AI58/snapshots/snapshots-2021-07-27/img-2021-07-27T17-37-21devID2.jpg
        # /dasio/AI58/masks/mask-id1.png
        camera_id = os.path.basename(photo_path)[28: -4]
        mask_path = os.path.join(os.path.dirname(photo_path), '..', '..', 'masks', f'mask-id{camera_id}.png')
        mask_path = os.path.abspath(mask_path)

        if mask_path not in self.mask_dict.keys():
            self.load_mask(mask_path)

        return self.mask_dict[mask_path]

    @staticmethod
    def imread(path):
        """reads using cv2 and converts to RGB"""
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_mask(self, mask_path):
        image = self.imread(mask_path)
        image = cv2.resize(image, self.output_size)
        self.mask_dict[mask_path] = np.where(image[:, :, 0] > 0, True, False)

    def __iter__(self):
        while True:
            with self.lock:
                if self.init_count == 0:
                    self.shuffle_data()
                    # clean batch_data
                    self.batch_data = []
                    self.init_count = 1

            for obj_id in self.objects_id_generator:
                curr_data = self.get_data_by_id(obj_id)

                image = curr_data['image']
                mask = curr_data['mask']
                flux = curr_data['flux']

                image = cv2.resize(image, self.output_size)
                segmentation_maps = SegmentationMapsOnImage(mask, shape=image.shape)

                if self.do_augment:
                    image, segmentation_maps = self.augmentation_sequence(image=image, segmentation_maps=segmentation_maps)

                mask = segmentation_maps.draw()[0]
                mask = np.where(mask > 0, 1, 0).transpose(2, 0, 1)

                image = (image / 255.0).transpose(2, 0, 1)
                image = torch.from_numpy(image)
                image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
                image = image * mask

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.batch_data)) < self.batch_size:
                        self.batch_data.append(
                            (image, flux))

                    if len(self.batch_data) >= self.batch_size:
                        # resnet50 require input for 4-dimensional weight [64, 3, 7, 7]
                        batch_x = np.stack(([i[0] for i in self.batch_data]), axis=0)
                        batch_y = np.array([i[1] for i in self.batch_data]).reshape([-1, 1])

                        yield batch_x, batch_y
                        self.batch_data = []
