import os
import pandas as pd
from compute_features import load_masks
# some constants
databases_dir = r"E:\AI-58\clouds_databases"
photos_base_dir = r"E:\AI-58\clouds_photos"
rad_dir = r"E:\AI-58\clouds_radiation"
observations_table_dir = r"E:\AI-58\clouds_observations"
tolerance = pd.Timedelta("10sec")
process_number = 1

photos_dirs = list(os.path.join(photos_base_dir, i) for i in
                   [
                        "snapshots-2021-08-02",
                        # "snapshots-2021-08-05",
                        # "snapshots-2021-08-09",
                        # "snapshots-2021-08-13",
                        # "snapshots-2021-08-17",
                        # "snapshots-2021-08-21",
                    ]
                   )

masks = load_masks(
    list(os.path.join(photos_base_dir, i) for i in
         [
             'mask10.png',
             'mask20.png',
         ]
         )
)
