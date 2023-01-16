import torch
from torchvision import transforms


class Normalizer:
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    normalizer = transforms.Normalize(means, stds)

    inv_normalizer = transforms.Compose([
        transforms.Normalize((0., 0., 0.), [1 / i for i in stds]),
        transforms.Normalize([-i for i in means], (1., 1., 1.)),
    ])

    @classmethod
    def call(cls, input_tensor: torch.tensor, inverted: bool = False):
        if inverted:
            return cls.inv_normalizer(input_tensor)
        else:
            return cls.normalizer(input_tensor)
