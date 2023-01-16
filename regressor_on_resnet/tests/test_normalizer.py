from pathlib import Path
import numpy as np

from ..flux_dataset import FluxDataset
from ..metadata_loader import MetadataLoader
from ..normalizer import Normalizer


def test_normalizer():
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
                                    batch_fields=['images'])

    batch = dataset_generator.__iter__().__next__()
    batch.to_tensor()

    normalized_image = Normalizer.call(batch.images)

    assert np.all(np.isclose((0, 0, 0), normalized_image.mean(dim=(0, 2, 3)).numpy(), atol=2))
    assert np.all(np.isclose((1, 1, 1), normalized_image.std(dim=(0, 2, 3)).numpy(), atol=1))

    denormalized_image = Normalizer.call(normalized_image, inverted=True)

    assert np.all(np.isclose(batch.images.mean(dim=(0, 2, 3)).numpy(),
                             denormalized_image.mean(dim=(0, 2, 3)).numpy()))
    assert np.all(np.isclose(batch.images.std(dim=(0, 2, 3)).numpy(),
                             denormalized_image.std(dim=(0, 2, 3)).numpy()))
