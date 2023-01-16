from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from regressor_on_resnet.batch_factory import BatchFactory
from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.metadata_loader import MetadataLoader


def get_real_factory(do_augment):
    expedition_config_path = Path(__file__) \
                                 .parent / '../../cloud_applications_v2/expeditions_configs/ABP-42-config.json'
    expedition_config_path = expedition_config_path.absolute()
    metadata_loader = MetadataLoader((expedition_config_path.__str__(),),
                                     radiation_threshold=10,
                                     split=(0.6, 0.2, 0.2))
    print('metadata_loader: done')
    batch_size = 32
    output_size = (512, 512)
    dataset_generator = FluxDataset(flux_frame=metadata_loader.all_df,
                                    batch_size=batch_size,
                                    do_shuffle=True,
                                    output_size=output_size,
                                    batch_fields=['images',
                                                  'masks',
                                                  'elevations'])
    print('dataset_generator: done')
    batch_factory = BatchFactory(dataset=dataset_generator,
                                 cuda_device=torch.device(0),
                                 do_augment=do_augment,
                                 cpu_queue_length=4,
                                 cuda_queue_length=1,
                                 preprocess_worker_number=20,
                                 cuda_feeder_number=1,
                                 to_variable=True)
    print('batch_factory: started')
    return batch_factory


@pytest.mark.parametrize("do_augment", [(True,), (False,)])
def test_batch_factory_real(capsys, do_augment):
    with capsys.disabled():
        batch_factory = get_real_factory(do_augment)
        keyboard_interruption = False

        try:
            batch = batch_factory.cuda_queue.get(block=True, timeout=60)
            print('batch: got')

            # check that normalization were done
            assert batch.images.abs().max().item() > 1

            elevations_max = batch.elevations.max().item()
            # angles must be in radians
            # catches angles in degrees and sin(elevation)
            assert 1 < elevations_max <= np.pi / 2

            # check mask apply
            # weak validation as zeros may occur on augmentation
            zeros_number = torch.where(batch.images == 0, 1, 0).sum()
            assert zeros_number >= torch.where(batch.masks == 0, 1, 0).sum()
            assert zeros_number >= 100

        except KeyboardInterrupt as e:
            print(e)
            keyboard_interruption = True
        finally:
            batch_factory.stop()

        assert not keyboard_interruption


def create_image(root_path: Path) -> Path:
    # /dasio/AI58/snapshots/snapshots-2021-07-27/img-2021-07-27T17-37-21devID2.jpg
    # /dasio/AI58/masks/mask-id1.png

    image = np.ones((1920, 1920, 3)) * 127
    image_path = root_path / 'snapshots/snapshots-2000-01-29/img-2000-01-29T12-00-00devID1.jpg'
    image_path.parent.mkdir(parents=True, )
    cv2.imwrite(image_path.__str__(), image)
    return image_path


def create_mask(root_path: Path, value: int):
    mask = np.ones((1920, 1920, 3)) * value
    mask_path = root_path / 'masks/mask-id1.png'
    mask_path.parent.mkdir(parents=True, )
    cv2.imwrite(mask_path.__str__(), mask)


def test_factory_synthetic_no_augment(tmp_path: Path):
    image_path = create_image(tmp_path)
    create_mask(tmp_path, 0)

    df = pd.DataFrame([{
        'photo_path': image_path.__str__(),
        'CM3up[W/m2]': 10,
        'sun_altitude': 45,
    }])
    batch_size = 10
    output_size = (512, 512)
    dataset_generator = FluxDataset(flux_frame=df,
                                    batch_size=batch_size,
                                    do_shuffle=True,
                                    output_size=output_size,
                                    batch_fields=['images',
                                                  'masks',
                                                  'elevations'])
    print('dataset_generator: done')
    batch_factory = BatchFactory(dataset=dataset_generator,
                                 cuda_device=torch.device(0),
                                 do_augment=False,
                                 cpu_queue_length=4,
                                 cuda_queue_length=1,
                                 preprocess_worker_number=20,
                                 cuda_feeder_number=1,
                                 to_variable=True)
    print('batch_factory: started')
    keyboard_interruption = False
    try:
        batch = batch_factory.cuda_queue.get(block=True, timeout=60)
        print('batch: got')

        # check mask apply
        zeros_number = torch.where(batch.images == 0, 1, 0).sum()
        assert zeros_number == np.product(batch.images.shape)

    except KeyboardInterrupt as e:
        print(e)
        keyboard_interruption = True
    finally:
        batch_factory.stop()

    assert not keyboard_interruption


def test_factory_synthetic_augment(tmp_path: Path):
    image_path = create_image(tmp_path)
    create_mask(tmp_path, 255)

    df = pd.DataFrame([{
        'photo_path': image_path.__str__(),
        'CM3up[W/m2]': 10,
        'sun_altitude': 45,
    }])
    batch_size = 1
    output_size = (512, 512)
    dataset_generator = FluxDataset(flux_frame=df,
                                    batch_size=batch_size,
                                    do_shuffle=True,
                                    output_size=output_size,
                                    batch_fields=['images',
                                                  'masks',
                                                  'elevations'])
    print('dataset_generator: done')
    batch_factory = BatchFactory(dataset=dataset_generator,
                                 cuda_device=torch.device(0),
                                 do_augment=True,
                                 cpu_queue_length=4,
                                 cuda_queue_length=1,
                                 preprocess_worker_number=20,
                                 cuda_feeder_number=1,
                                 to_variable=True)
    print('batch_factory: started')
    keyboard_interruption = False
    corners = []
    try:
        for i in range(50):
            batch = batch_factory.cuda_queue.get(block=True, timeout=60)
            # check augmentation
            corners.append(batch.images[:, :, 0, 0])

        assert torch.where(torch.stack(corners) == 0, 1, 0).sum().item() > 50 / 2 * 3

    except KeyboardInterrupt as e:
        print(e)
        keyboard_interruption = True
    finally:
        batch_factory.stop()

    assert not keyboard_interruption
