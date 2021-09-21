import threading
import accimage
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from sklearn.utils import shuffle
import os
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import cv2





class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_objects_i(objects_count):
    """Cyclic generator of paths indices
    """
    current_objects_id = 0
    while True:
        yield current_objects_id
        current_objects_id = (current_objects_id + 1) % objects_count


class SunLandmarksDataset(Dataset):
    """Sun Landmarks dataset."""

    def __init__(self, csv_file, root_dir, batch_size, resize = 512):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, delimiter=',')
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.resize = resize

        self.objects_id_generator = threadsafe_iter(get_objects_i(self.landmarks_frame.shape[0]))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0
        self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                   iaa.Flipud(0.5),
                                   iaa.GaussianBlur(sigma=(0, 5)),
                                   iaa.Affine(rotate=(-180,180),
                                              translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                                              shear={'x': (-10, 10), 'y': (-10, 10)})])
        self.cache = {}

    def __len__(self):
        return len(self.landmarks_frame)

    def shuffle(self):
        self.landmarks_frame = shuffle(self.landmarks_frame)

    def __iter__(self):
        while True:
            with self.lock:
                if (self.init_count == 0):
                    self.shuffle()
                    self.imgs = []
                    self.landmarks = []
                    self.init_count = 1

            for obj_id in self.objects_id_generator:
                img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[obj_id, 0])
                if img_name in self.cache:
                    img = self.cache[img_name]
                else:
                    img = accimage.Image(img_name)
                    img = img.resize((self.resize,self.resize))
                    image_np = np.empty([img.channels, img.height, img.width], dtype=np.uint8)      # CxHxW
                    img.copyto(image_np)
                    self.cache[img_name] = img
                img = np.transpose(image_np, (1, 2, 0)).astype(np.uint8)                            # HxWxC

                landmarks = self.landmarks_frame.iloc[obj_id, 1:]
                landmarks = np.array([landmarks])*self.resize
                landmarks = landmarks.astype('float').reshape(3,)

                R = landmarks[2]
                kps = KeypointsOnImage([Keypoint(x=landmarks[0], y=landmarks[1])], shape=img.shape)
                image_aug, landmarks_after = self.seq(image=img, keypoints=kps)                     # тут применяю аугментацию к фотке и целевой переменной
                landmarks[0] = landmarks_after.keypoints[0].x
                landmarks[1] = landmarks_after.keypoints[0].y
                img = image_aug

                img = img.transpose((2, 0, 1))[np.newaxis, ...]                                     # 1 x CxHxW
                landmarks = landmarks[np.newaxis, ...]                                              # 1 x 3

                ###### Coords only - without radius ######
                # landmarks = landmarks[:, :2]                                                      # 1 x 2
                ###### Coords only - without radius ######

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.imgs)) < self.batch_size:
                        self.imgs.append(img)
                        self.landmarks.append(landmarks)
                    if len(self.imgs) % self.batch_size == 0:
                        self.imgs = np.concatenate(self.imgs, axis=0)
                        self.landmarks = np.concatenate(self.landmarks, axis=0)
                        yield (self.imgs, self.landmarks)
                        self.imgs, self.landmarks = [], []

            # At the end of an epoch we re-init data-structures
            with self.lock:
                self.landmarks_frame = shuffle(self.landmarks_frame)
                self.init_count = 0






class SunDiskDataset(Dataset):
    """Sun Landmarks dataset."""

    def __init__(self, csv_file, root_dir, batch_size, resize = 512, augment = True, shuffle = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, delimiter=',')
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.resize = resize
        self.augment = augment
        self.shuffle = shuffle

        self.objects_id_generator = threadsafe_iter(get_objects_i(self.landmarks_frame.shape[0]))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0
        if self.augment:
            self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                       iaa.Flipud(0.5),
                                       iaa.GaussianBlur(sigma=(0, 5)),
                                       # iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False),
                                       iaa.Affine(rotate=(-180,180),
                                                  translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                                                  shear={'x': (-10, 10), 'y': (-10, 10)})],
                                      random_order=True)
            self.img_aug = iaa.Sequential([iaa.Cutout(nb_iterations=(1, 3), size=0.3, squared=False, cval=0),
                                           iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))])
        else:
            self.seq = iaa.Identity()
            self.img_aug = iaa.Identity()
        self.cache = {}

    def __len__(self):
        return len(self.landmarks_frame)

    def shuffle_data(self):
        self.landmarks_frame = shuffle(self.landmarks_frame)

    def __iter__(self):
        while True:
            with self.lock:
                if (self.init_count == 0):
                    if self.shuffle:
                        self.shuffle_data()
                    self.imgs = []
                    self.segmaps = []
                    self.init_count = 1

            for obj_id in self.objects_id_generator:
                img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[obj_id, 0])
                if img_name in self.cache:
                    img = self.cache[img_name]
                else:
                    img = accimage.Image(img_name)
                    img = img.resize((self.resize,self.resize))
                    image_np = np.empty([img.channels, img.height, img.width], dtype=np.uint8)      # CxHxW
                    img.copyto(image_np)
                    self.cache[img_name] = img
                img = np.transpose(image_np, (1, 2, 0)).astype(np.uint8)                            # HxWxC

                landmarks = self.landmarks_frame.iloc[obj_id, 1:]
                landmarks = np.array([landmarks])*self.resize
                landmarks = landmarks.astype('float').reshape(3,)
                x, y, r = landmarks
                x, y, r = int(x), int(y), int(r)
                sundisk_img = np.zeros(img.shape, dtype=np.uint8)
                sundisk_segmap = cv2.circle(sundisk_img, (x, y), r, (255, 255, 255), thickness=-1)

                sundisk_segmap = sundisk_segmap[:,:,0][np.newaxis, :,:, np.newaxis]

                img, sundisk_segmap = self.seq(image=img, segmentation_maps=sundisk_segmap)
                img = self.img_aug(image=img)
                sundisk_segmap = sundisk_segmap[0,:,:,:]
                sundisk_segmap = (sundisk_segmap >= 128).astype(np.float32)
                sundisk_segmap = sundisk_segmap + np.random.randn(*(sundisk_segmap.shape))/500.0
                sundisk_segmap = np.clip(sundisk_segmap, 0.0, 1.0)
                sundisk_segmap = np.transpose(sundisk_segmap, (2,0,1))[np.newaxis, ...]             # 1 x 1xHxW

                img = img.transpose((2, 0, 1))[np.newaxis, ...]                                     # 1 x CxHxW

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if (len(self.imgs)) < self.batch_size:
                        self.imgs.append(img)
                        self.segmaps.append(sundisk_segmap)
                    if len(self.imgs) % self.batch_size == 0:
                        self.imgs = np.concatenate(self.imgs, axis=0)
                        self.segmaps = np.concatenate(self.segmaps, axis=0)
                        yield (self.imgs, self.segmaps)
                        self.imgs, self.segmaps = [], []

            # At the end of an epoch we re-init data-structures
            with self.lock:
                self.init_count = 0
                if self.shuffle:
                    self.landmarks_frame = shuffle(self.landmarks_frame)




class SunDiskDataset_iterable(Dataset):
    """Sun Landmarks dataset."""

    def __init__(self, csv_file, root_dir, resize = 512, augment = True, shuffle = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, delimiter=',')
        self.root_dir = root_dir
        self.resize = resize
        self.augment = augment
        self.shuffle = shuffle

        self.objects_id_generator = threadsafe_iter(get_objects_i(self.landmarks_frame.shape[0]))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0
        if self.augment:
            self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                                       iaa.Flipud(0.5),
                                       # iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False),
                                       iaa.Affine(rotate=(-180,180),
                                                  translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                                                  shear={'x': (-10, 10), 'y': (-10, 10)}),
                                       iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)],
                                      random_order=True)
            self.img_aug = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 5)),
                                           iaa.Cutout(nb_iterations=(1, 3), size=0.3, squared=False, cval=0),
                                           iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))])
        else:
            self.seq = iaa.Identity()
            self.img_aug = iaa.Identity()
        self.cache = {}

    def __len__(self):
        return len(self.landmarks_frame)

    def shuffle_data(self):
        self.landmarks_frame = shuffle(self.landmarks_frame)

    def __getitem__(self, obj_id):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[obj_id, 0])
        if img_name in self.cache:
            img = self.cache[img_name]
        else:
            # img = cv2.imread(img_name)
            img = accimage.Image(img_name)
            img = img.resize((self.resize, self.resize))
            image_np = np.empty([img.channels, img.height, img.width], dtype=np.uint8)      # CxHxW
            img.copyto(image_np)
            self.cache[img_name] = img
        img = np.transpose(image_np, (1, 2, 0)).astype(np.uint8)                            # HxWxC

        landmarks = self.landmarks_frame.iloc[obj_id, 1:]
        landmarks = np.array([landmarks])*self.resize
        landmarks = landmarks.astype('float').reshape(3,)
        x, y, r = landmarks
        x, y, r = int(x), int(y), int(r)
        sundisk_img = np.zeros(img.shape, dtype=np.uint8)
        sundisk_segmap = cv2.circle(sundisk_img, (x, y), r, (255, 255, 255), thickness=-1)

        sundisk_segmap = sundisk_segmap[:,:,0][np.newaxis, :,:, np.newaxis]

        img, sundisk_segmap = self.seq(image=img, segmentation_maps=sundisk_segmap)
        img = self.img_aug(image=img)
        sundisk_segmap = sundisk_segmap[0,:,:,:]
        sundisk_segmap = (sundisk_segmap >= 128).astype(np.float32)
        sundisk_segmap = sundisk_segmap + np.random.randn(*(sundisk_segmap.shape))/500.0
        sundisk_segmap = np.clip(sundisk_segmap, 0.0, 1.0)
        sundisk_segmap = np.transpose(sundisk_segmap, (2,0,1))

        img = img.transpose((2, 0, 1))

        return img, sundisk_segmap