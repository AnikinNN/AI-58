from pathlib import Path

import pandas as pd

from cloud_lib_v2.photos_processing import init_events


def create_fake_photos(base_path: Path):
    created_photos = []
    for day in range(1, 6):
        day_path = base_path / f'snapshots-2000-07-{day:0>2d}'
        day_path.mkdir()
        for hour in range(10):
            for camera_id in range(2):
                photo_path = day_path / f'img-2000-07-{day:0>2d}T{hour:0>2d}-42-20devID{camera_id}.jpg'
                with open(photo_path, 'w'):
                    pass
                created_photos.append(photo_path)
    return created_photos


def test_init_events(tmp_path: Path):
    created_photos = create_fake_photos(tmp_path)
    df_events = init_events(tmp_path.__str__())

    assert isinstance(df_events, pd.DataFrame)
    assert set(i.__str__() for i in created_photos) == set(df_events['photo_path'])
    assert {'0', '1'} == set(df_events['camera_id'])
