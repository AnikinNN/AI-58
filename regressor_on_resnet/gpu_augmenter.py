import numpy as np
import torch
from torchvision import transforms

from regressor_on_resnet.batch import FluxBatch


class Augmenter:
    normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    inv_normalizer = transforms.Compose([
        transforms.Normalize((0., 0., 0.), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.)),
    ])

    class Sampler:
        def __init__(self):
            self.rotation_angle = self.sampler(5)
            self.shear_x = self.sampler(5)
            self.shear_y = self.sampler(5)
            self.flip_ud = self.sampler(0.5, 0.5)
            self.flip_lr = self.sampler(0.5, 0.5)
            self.brightness_multiplier = self.sampler(0.02, 1)
            self.brightness_delta = self.sampler(0.02)

        @staticmethod
        def sampler(delta: float = 1, center: float = 0):
            return (np.random.rand() - 0.5) * 2 * delta + center

    @classmethod
    def augment_image(cls, images: torch.Tensor, is_mask: bool, sampler: Sampler):

        images = transforms.functional.affine(images, sampler.rotation_angle, [0, 0], 1,
                                              [sampler.shear_x, sampler.shear_y],
                                              interpolation=transforms.InterpolationMode.NEAREST)

        if sampler.flip_ud > 0.5:
            images = torch.flip(images, dims=[2])
        if sampler.flip_lr > 0.5:
            images = torch.flip(images, dims=[3])

        if not is_mask:
            images = transforms.functional.adjust_brightness(images, sampler.brightness_multiplier)
            images = (images + sampler.brightness_delta).clamp(0, 1.0)
        return images

    @classmethod
    def augment_elevation(cls, elevation: torch.Tensor):
        elevation_augmented = torch.clip(torch.normal(elevation, 1.0), -90, 90)
        return elevation_augmented

    @classmethod
    def __call__(cls, batch: FluxBatch):
        sampler = cls.Sampler()
        images = cls.augment_image(batch.images, False, sampler)
        masks = cls.augment_image(batch.masks, True, sampler)
        elevations = cls.augment_elevation(batch.elevations)
        return images, masks, elevations

    @classmethod
    def call(cls, batch: FluxBatch):
        return cls.__call__(batch)
