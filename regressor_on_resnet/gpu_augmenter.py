import numpy as np
import torch
from torchvision import transforms

from regressor_on_resnet.flux_batch import FluxBatch


class Augmenter:
    normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    inv_normalizer = transforms.Compose([
        transforms.Normalize((0., 0., 0.), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.)),
    ])

    @staticmethod
    def sampler(delta: float = 1, center: float = 0):
        return (np.random.rand() - 0.5) * 2 * delta + center

    @classmethod
    def augment(cls, images: torch.Tensor, is_mask: bool):

        images = transforms.functional.affine(images, cls.sampler(45), [0, 0], 1,
                                              [cls.sampler(5), cls.sampler(5)],
                                              interpolation=transforms.InterpolationMode.NEAREST)

        if cls.sampler(0.5, 0.5) > 0.5:
            images = torch.flip(images, dims=[1])
        if cls.sampler(0.5, 0.5) > 0.5:
            images = torch.flip(images, dims=[2])

        if not is_mask:
            images = transforms.functional.adjust_brightness(images, cls.sampler(0.15, 1))
            images = (images + cls.sampler(0.2)).clamp(0, 1.0)
            images = cls.normalizer(images)
        return images

    @classmethod
    def __call__(cls, batch: FluxBatch):
        images = cls.augment(batch.images, False)
        masks = cls.augment(batch.masks, True)
        return images * masks

    @classmethod
    def call(cls, batch: FluxBatch):
        return cls.__call__(batch)
