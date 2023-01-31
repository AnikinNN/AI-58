import os
import cv2
import pandas as pd
import threading
import numpy as np
from sklearn.utils import shuffle

from regressor_on_resnet.batch import FluxBatch
from regressor_on_resnet.threadsafe_iterator import ThreadsafeIterator


def get_object_index(objects_count):
    """
    Cyclic generator of indices from 0 to objects_count
    """
    current_id = 0
    while True:
        yield current_id
        current_id = (current_id + 1) % objects_count


class FluxDataset:
    def __init__(self,
                 flux_frame: pd.DataFrame,
                 batch_size: int,
                 do_shuffle: bool,
                 output_size: tuple[int, int],
                 batch_fields: list[str, ...]):
        self.flux_frame = flux_frame
        self.do_shuffle = do_shuffle
        self.batch_size = batch_size
        self.output_size = output_size
        self.batch_fields = batch_fields

        self.objects_iloc_generator = ThreadsafeIterator(get_object_index(self.flux_frame.shape[0]))
        self.yield_lock = threading.Lock()

        self.batch: FluxBatch = None
        self.clean_batch()
        self.mask_dict = {}

        if self.do_shuffle:
            self.shuffle_data()

    def __len__(self):
        return self.flux_frame.shape[0]

    def shuffle_data(self):
        if self.do_shuffle:
            self.flux_frame = shuffle(self.flux_frame)

    def get_data_by_id(self, index):
        result = {}
        row = self.flux_frame.iloc[index]
        photo_path = row['photo_path']

        for field in self.batch_fields:
            if field == 'images':
                result[field] = self.get_image(photo_path)
            elif field == 'masks':
                result[field] = self.get_mask(photo_path)
            elif field == 'fluxes':
                result[field] = row['CM3up[W/m2]']
            elif field == 'elevations':
                result[field] = row['sun_altitude']
            elif field == 'train_frame_indexes':
                result[field] = row.name
            elif field == 'hard_mining_weights':
                result[field] = row['hard_mining_weight']
            elif field == 'true_radiation_class':
                result[field] = row['radiation_class_oh']
            else:
                raise ValueError(f'got {field=} which is unsupported')

        return result

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

    def get_image(self, path):
        image = self.imread(path)
        image = self.resize(image)
        # make a torch style image
        image = (image / 255.).transpose(2, 0, 1)
        return image

    def load_mask(self, mask_path):
        # read in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.resize(mask)
        # convert to mask with 0 and 1 to use it as multiplier to an actual image
        mask = np.where(mask > 0, 1., 0.)
        # make it 3d along new axis and store
        self.mask_dict[mask_path] = np.repeat(mask[np.newaxis, :, :], 3, axis=0)

    def __iter__(self):
        while True:
            for obj_iloc in self.objects_iloc_generator:
                batch_appendix = self.get_data_by_id(obj_iloc)

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if len(self.batch) < self.batch_size:
                        self.batch.append(**batch_appendix)

                    if len(self.batch) >= self.batch_size:
                        yield self.batch
                        self.clean_batch()

    def clean_batch(self):
        self.batch = FluxBatch()

    def resize(self, image):
        result = cv2.resize(image, self.output_size)
        return result
