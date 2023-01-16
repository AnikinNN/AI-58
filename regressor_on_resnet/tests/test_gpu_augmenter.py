from pathlib import Path

import numpy as np
import torch

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.gpu_augmenter import Augmenter
from regressor_on_resnet.metadata_loader import MetadataLoader


def test_augment_image():
    image = torch.ones(32, 3, 512, 512) * 0.01

    sampler = Augmenter.Sampler()
    sampler.rotation_angle = 90
    sampler.shear_x = 0
    sampler.shear_y = 0
    sampler.brightness_delta = 0.02
    sampler.brightness_multiplier = 10
    expected_brightness = 0.01 * 10 + 0.02

    augmented_image = Augmenter.augment_image(image, is_mask=False, sampler=sampler)
    assert np.isclose(augmented_image.mean().item(), expected_brightness)


def test_augment_elevation():
    elevations = torch.normal(45, 5, (32, 1))
    elevations = Augmenter.augment_elevation(elevations)

    elevations_mean = elevations.mean().item()
    assert np.isclose(elevations_mean, 45, atol=3)

    assert elevations.max().item() <= 90
    assert elevations.min().item() >= -90


def test_call():
    expedition_config_path = Path(__file__).parent / '../../cloud_applications_v2/expeditions_configs/AI-58-config.json'
    expedition_config_path = expedition_config_path.absolute()
    metadata_loader = MetadataLoader((expedition_config_path.__str__(),),
                                     radiation_threshold=10,
                                     split=(0.6, 0.2, 0.2))
    batch_size = 128
    output_size = (512, 512)
    dataset_generator = FluxDataset(flux_frame=metadata_loader.all_df,
                                    batch_size=batch_size,
                                    do_shuffle=True,
                                    output_size=output_size,
                                    batch_fields=['images',
                                                  'masks',
                                                  'elevations'])

    batch = dataset_generator.__iter__().__next__()
    batch.to_tensor()

    images, masks, elevations = Augmenter.call(batch)

    # check that augmentations didn't change image's range
    assert images.max().item() <= 1
    assert images.min().item() >= 0

    # check that there is only 0 and 1 values in masks
    assert np.array_equal(np.unique(masks.numpy()), np.array([0, 1]))

    elevations_max = elevations.max().item()
    # angles must be in degrees
    # catches angles in radians and sin(elevation)
    assert np.pi / 2 < elevations_max <= 90
