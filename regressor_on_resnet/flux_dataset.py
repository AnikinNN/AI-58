import os
import cv2
import torch
import threading
import numpy as np

from sklearn.utils import shuffle
from torchvision.transforms import transforms
from imgaug import SegmentationMapsOnImage, augmenters

from regressor_on_resnet.threadsafe_iterator import ThreadsafeIterator


def get_object_index(objects_count):
    """Cyclic generator of indices from 0 to objects_count
    """
    current_id = 0
    while True:
        yield current_id
        current_id = (current_id + 1) % objects_count


class FluxDataset:
    normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    inv_normalizer = transforms.Compose([
        transforms.Normalize((0., 0., 0.), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.)),
    ])

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
            augmenters.Fliplr(0.5),
            augmenters.Flipud(0.5),
            # augmenters.Dropout([0.05, 0.2]),
            augmenters.Affine(shear=(-16, 16), rotate=(-45, 45)),
            augmenters.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
        ], random_order=True)

        self.output_size = (512, 512)

        self.batch_x = None
        self.batch_y = None

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
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = self.resize(image)
        self.mask_dict[mask_path] = np.where(image > 0, True, False)

    def __iter__(self):
        while True:
            with self.lock:
                if self.init_count == 0:
                    self.shuffle_data()
                    # clean batch_data
                    self.clean_batch()
                    self.init_count = 1

            for obj_id in self.objects_id_generator:
                curr_data = self.get_data_by_id(obj_id)

                image = curr_data['image']
                mask = curr_data['mask']
                flux = curr_data['flux']

                image = self.resize(image)

                if self.do_augment:
                    segmentation_maps = SegmentationMapsOnImage(mask, shape=image.shape)
                    image, segmentation_maps = self.augmentation_sequence(image=image,
                                                                          segmentation_maps=segmentation_maps)
                    mask = segmentation_maps.draw()[0]
                    mask = np.where(mask > 0, 1, 0).transpose(2, 0, 1)
                else:
                    pass

                image = (image / 255.0).transpose(2, 0, 1)
                image = torch.from_numpy(image)
                image = self.normalizer(image)

                image = image * mask

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if len(self.batch_y) < self.batch_size:
                        # resnet50 require input for 4-dimensional weight [64, 3, 7, 7]
                        self.batch_x.append(image)
                        self.batch_y.append(flux)

                    if len(self.batch_y) >= self.batch_size:
                        yield self.batch_x, self.batch_y
                        self.clean_batch()

    def clean_batch(self):
        self.batch_x = []
        self.batch_y = []

    def resize(self, image):
        result = cv2.resize(image, self.output_size)
        return result
