import cv2
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from ..batch import FluxBatch
from ..flux_dataset import FluxDataset
from ..metadata_loader import MetadataLoader


def create_image(root_path: Path) -> Path:
    # /dasio/AI58/snapshots/snapshots-2021-07-27/img-2021-07-27T17-37-21devID2.jpg
    # /dasio/AI58/masks/mask-id1.png

    image = np.ones((1920, 1920, 3)) * 127
    image_path = root_path / 'snapshots/snapshots-2000-01-29/img-2000-01-29T12-00-00devID1.jpg'
    image_path.parent.mkdir(parents=True, )
    cv2.imwrite(image_path.__str__(), image)

    mask = np.ones((1920, 1920, 3)) * 255
    mask[:960, :, :] = 0
    mask_path = root_path / 'masks/mask-id1.png'
    mask_path.parent.mkdir(parents=True, )
    cv2.imwrite(mask_path.__str__(), mask)

    return image_path


def test_dataset_generator_synthetic(tmp_path: Path):
    image_path = create_image(tmp_path)
    df = pd.DataFrame([{
        'photo_path': image_path.__str__(),
        'CM3up[W/m2]': 10,
        'sun_altitude': 45,
    }])
    dataset_generator = FluxDataset(flux_frame=df,
                                    batch_size=32,
                                    do_shuffle=True,
                                    output_size=(512, 512),
                                    batch_fields=['images',
                                                  'masks',
                                                  'fluxes',
                                                  'elevations',
                                                  'train_frame_indexes', ])

    batch = dataset_generator.__iter__().__next__()

    assert isinstance(batch, FluxBatch)
    assert len(batch) == 32

    batch.to_tensor()
    assert np.isclose(batch.images.mean().item(), 127 / 255)
    assert np.isclose(batch.masks.mean().item(), 0.5)

    assert np.isclose((batch.images * batch.masks).mean().item(), 127 / 2 / 255)

    assert batch.images.shape == batch.masks.shape
    assert batch.images.shape == (32, 3, 512, 512)

    assert batch.elevations.shape == (32, 1)
    assert batch.fluxes.shape == (32, 1)

    assert batch.elevations[0, 0] == 45
    assert batch.fluxes[0, 0] == 10


def test_dataset_generator_real():
    expedition_config_path = Path(__file__).parent / '../../cloud_applications_v2/expeditions_configs/AI-58-config.json'
    expedition_config_path = expedition_config_path.absolute()
    metadata_loader = MetadataLoader((expedition_config_path.__str__(),),
                                     radiation_threshold=10,
                                     split=(0.6, 0.2, 0.2))
    batch_size = 32
    output_size = (512, 512)
    dataset_generator = FluxDataset(flux_frame=metadata_loader.all_df,
                                    batch_size=batch_size,
                                    do_shuffle=True,
                                    output_size=output_size,
                                    batch_fields=['images',
                                                  'masks',
                                                  'fluxes',
                                                  'elevations',
                                                  'train_frame_indexes', ])

    batch = dataset_generator.__iter__().__next__()
    batch.to_tensor()

    # check that image values in [0, 1] range
    assert torch.all(torch.less_equal(torch.greater_equal(batch.images, 0), 1)).item()

    # check that there is only 0 and 1 values in batch.masks
    assert np.array_equal(np.unique(batch.masks.numpy()), np.array([0, 1]))

    elevations_max = batch.elevations.max().item()
    # angles must be in degrees
    # catches angles in radians and sin(elevation)
    assert np.pi / 2 < elevations_max <= 90

    # check shapes
    assert batch.images.shape == batch.masks.shape
    assert batch.images.shape == (batch_size, 3, *output_size)


@pytest.mark.parametrize("do_shuffle, threshold", [(True, 5), (False, 10)])
def test_shuffle_data(do_shuffle, threshold):
    """
    This test includes statistical assumption
    """
    flux_frame = pd.DataFrame(np.linspace(0, 1, 200).reshape(100, 2))
    shuffle_result = []

    for i in range(10):
        dataset_generator = FluxDataset(flux_frame=flux_frame,
                                        batch_size=32,
                                        do_shuffle=do_shuffle,
                                        output_size=(10, 10),
                                        batch_fields=[])
        row = dataset_generator.flux_frame.iloc[0]
        shuffle_result.append(row.name)

    shuffle_result = sum([i == 0 for i in shuffle_result])

    if do_shuffle:
        assert shuffle_result <= threshold
    else:
        assert shuffle_result == threshold
