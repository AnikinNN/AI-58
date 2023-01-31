from pathlib import Path

import numpy as np
import torch

from regressor_on_resnet.metadata_loader import MetadataLoader

this_script_path = Path(__file__)
project_root_path = this_script_path.parent.parent.parent


def test_init_radiation_classes():
    metadata_loader = MetadataLoader((
        project_root_path / 'cloud_applications_v2/expeditions_configs/AI-49-config.json',
    ),
        radiation_class_number=8)

    assert 'radiation_class' in metadata_loader.all_df.columns
    assert 'radiation_class_oh' in metadata_loader.all_df.columns

    assert np.all(np.diff(metadata_loader.radiation_percentile) > 0)

    # check that classes contain the same amount of events
    counts = metadata_loader.all_df.radiation_class.value_counts().sort_index().to_numpy()
    counts = counts / len(metadata_loader.all_df)
    assert np.all(np.isclose(counts, 1 / 8, atol=0.02))

    # check type and shape of one_hot_encoding
    assert isinstance(metadata_loader.all_df.iloc[0].radiation_class_oh, np.ndarray)
    assert metadata_loader.all_df.iloc[0].radiation_class_oh.shape == (8,)
    assert np.unique(np.stack(metadata_loader.all_df.radiation_class_oh).sum(axis=1)) == np.ones(1)

    # check similarity of amount ohe and classes
    counts_oh = np.stack(metadata_loader.all_df.radiation_class_oh).sum(axis=0)
    counts_oh = counts_oh / len(metadata_loader.all_df)
    assert np.all(counts_oh == counts)

    # check radiation_class's range is an interval [0, 8)
    assert set(i for i in range(8)) == set(metadata_loader.all_df.radiation_class.value_counts().index)

    # check some of one_hot_encodins
    for _ in range(100):
        row = metadata_loader.all_df.sample(n=1, axis='index').iloc[0]
        ohe = np.zeros(8)
        ohe[row.radiation_class] = 1

        assert np.all(ohe == row.radiation_class_oh)


def test_store_metadata(tmpdir: Path):
    metadata_loader = MetadataLoader(
        (project_root_path / 'cloud_applications_v2/expeditions_configs/AI-49-config.json',),
        radiation_class_number=8,
        store_path=tmpdir
    )

    for name in ('train', 'test', 'validation'):
        assert (tmpdir / f'subset_{name}.csv').exists()

    npy_path = (tmpdir / 'radiation_percentile.npy')
    assert npy_path.exists()

    npy = np.load(npy_path.__str__())
    assert npy.shape == (8, )
