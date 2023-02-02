from pathlib import Path
import pytest

from cloud_lib_v2.observation import init_observation

# ((table_path, shape), ...)
tables = [
    (Path('/dasio/AI60/AI60.xlsx'), (151, 29)),
    (Path('/dasio/AI61/AI61-observations.MB.xlsx'), (229, 16)),
    (Path('/dasio/AI45/AI45-short.xlsx'), (113, 11)),
    (Path('/dasio/AMK79/AMK79-short.xlsx'), (355, 11)),
    (Path('/dasio/AMK71/AMK71-short.xlsx'), (824, 11)),
    (Path('/dasio/AMK70/AMK70-short.xlsx'), (68, 11)),
    (Path('/dasio/ABP42/ABP42-short.xlsx'), (655, 11)),
    (Path('/dasio/AI52/AI52-short.xlsx'), (381, 11)),
    (Path('/dasio/AI49/AI49-short.xlsx'), (330, 11)),
    (Path('/dasio/ANS31/ANS31_short.xlsx'), (349, 11)),
]


@pytest.mark.parametrize("table_path, shape", tables)
def test_init_observation(table_path, shape):
    df = init_observation(table_path)

    assert df.shape == shape
    assert df['observation_datetime'].is_monotonic_increasing
    assert 'cloud_type' in df.columns
    assert 'TCC' in df.columns


def test_completeness():
    dasio_path = Path('/dasio')
    table_paths = set(dasio_path.glob('*/*.xlsx'))
    assert table_paths == set(i[0] for i in tables)
