from pathlib import Path
import sys
import numpy as np
import torch

from regressor_on_resnet.batch_factory import BatchFactory
from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.metadata_loader import MetadataLoader
from regressor_on_resnet.normalizer import Normalizer


def test_batch_factory(capsys):
    with capsys.disabled():
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
                                     do_augment=True,
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

            # check that normalization were done
            assert batch.images.abs().max().item() > 1

            elevations_max = batch.elevations.max().item()
            # angles must be in radians
            # catches angles in degrees and sin(elevation)
            assert 1 < elevations_max <= np.pi / 2
        except KeyboardInterrupt as e:
            print(e)
            keyboard_interruption = True
        finally:
            batch_factory.stop()

        assert not keyboard_interruption


def test_stop():
    assert False
