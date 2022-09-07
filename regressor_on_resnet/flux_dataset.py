import os
import cv2
import torch
import threading
import numpy as np

from sklearn.utils import shuffle
from torchvision.transforms import transforms
from imgaug import SegmentationMapsOnImage, augmenters

from regressor_on_resnet.flux_batch import FluxBatch
from regressor_on_resnet.threadsafe_iterator import ThreadsafeIterator


def get_object_index(objects_count):
    """Cyclic generator of indices from 0 to objects_count
    """
    current_id = 0
    while True:
        yield current_id
        current_id = (current_id + 1) % objects_count


class FluxDataset:
    def __init__(self, flux_frame, batch_size=32, do_shuffle=True):

        self.flux_frame = flux_frame
        self.mask_dict = {}

        self.do_shuffle = do_shuffle
        self.batch_size = batch_size

        self.objects_iloc_generator = ThreadsafeIterator(get_object_index(self.flux_frame.shape[0]))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0

        self.output_size = (512, 512)

        self.batch = FluxBatch()

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
        elevation = self.flux_frame.iloc[index]['sun_altitude']
        row_id = self.flux_frame.iloc[index].name
        hard_mining_weight = self.flux_frame.iloc[index]['hard_mining_weight']

        return image, mask, flux, elevation, row_id, hard_mining_weight

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
        # read in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.resize(mask)
        # convert to mask with 0 and 1 to use it as multiplier to an actual image
        mask = np.where(mask > 0, 1., 0.)
        # make it 3d along new axis and store
        self.mask_dict[mask_path] = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    def __iter__(self):
        while True:
            with self.lock:
                # todo implement initial shuffle without using self.init_count
                if self.init_count == 0:
                    self.shuffle_data()
                    # clean batch_data
                    self.clean_batch()
                    self.init_count = 1

            for obj_iloc in self.objects_iloc_generator:
                image, mask, flux, elevation, row_id, hard_mining_weight = self.get_data_by_id(obj_iloc)

                image = self.resize(image)

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if len(self.batch) < self.batch_size:
                        # resnet50 require input for 4-dimensional weight [64, 3, 7, 7]
                        self.batch.append(image, mask, elevation, flux, hard_mining_weight, row_id)

                    if len(self.batch) >= self.batch_size:
                        yield self.batch
                        self.clean_batch()

    def clean_batch(self):
        self.batch = FluxBatch()

    def resize(self, image):
        result = cv2.resize(image, self.output_size)
        return result
