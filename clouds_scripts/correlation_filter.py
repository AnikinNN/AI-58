import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from compute_features import *
from connect_rad_photos import *
from multiprocessing import Pool, freeze_support
from path import *

photos = pd.DataFrame(get_photo_names(photos_base_dir), columns=["photos"])
photos["photo_time"] = photos.apply(lambda row: extract_time(row["photos"]), axis=1)
photos["photo_time"] = pd.to_datetime(photos["photo_time"])

# delete unused photos or filter them
start_date = pd.to_datetime("2021-08-16 00:59:00")
end_date = pd.to_datetime("2021-08-16 1:10:00")
selection = (photos["photo_time"] > start_date) & (photos["photo_time"] < end_date)
# apply selection
photos = photos[selection]

photos = photos.reset_index(drop=True)

pbar = tqdm(total=photos.shape[0], position=0, leave=True)


def correlation(a, b):
    return np.corrcoef(a, b)[0, 1]


def correlation_to_previous(row: pd.Series):
    # find next photo
    prev_img = None
    prev_index = row.name
    while prev_img is None and prev_index > 0:
        prev_index -= 1
        if photos.iloc[prev_index]["photos"][-5] == row["photos"][-5]:
            prev_img = get_masked_image(get_full_path(photos.iloc[prev_index]["photos"]), masks)

    current_img = get_masked_image(get_full_path(row["photos"]), masks)
    corr = correlation(current_img[5], prev_img[5]) if prev_img is not None else None
    pbar.update()
    return corr


photos["corr_to_prev"] = photos.apply(correlation_to_previous, axis=1)
photos["is_anomaly"] = photos.apply(lambda row: row["corr_to_prev"] < anomaly_threshold, axis=1)
photos.to_csv(os.path.join(databases_dir, "anomalies.csv"), index=False)

plt.plot(photos["photo_time"], photos["corr_to_prev"])

plt.show()
