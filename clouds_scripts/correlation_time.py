import re
import matplotlib.pyplot as plt
from scipy import stats
from numpy import ma
import imageio
import os
import pandas as pd
import numpy as np

from compute_features import resize4x, HSV, get_masked_image
from connect_rad_photos import extract_time, get_full_path
from service_defs import find_files

from path import *

# correlation_thresholds must be sorted
correlation_thresholds = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# read table with observations
observations = pd.DataFrame()
for file in find_files(observations_table_dir, "AI58_Clouds*.xlsx"):
    observations = observations.append(pd.read_excel(file))

# drop rows that doesnt contain SDS, because SDS is easiest feature to determine
# drop rows that doesnt contain Date_Time, because they doesnt contain anything
observations = observations.dropna(how="any", axis="index", subset=["SDS", "Date_Time"])
# delete headers
observations = observations[observations["TCC"] != "TCC"]
# keep rows with clouds by TCC
observations = observations[observations["TCC"] != "-"]


# capitalize "п" to achieve sameness
def capitalize_sds(row: pd.Series):
    if type(row["SDS"]) == str:
        row["SDS"] = row["SDS"].capitalize()
    return row


observations = observations.apply(capitalize_sds, axis=1)

# cast to datetime
observations["Date_Time"] = pd.to_datetime(observations["Date_Time"])

# read all photos
photos = []
for subdir in os.listdir(photos_base_dir):
    if re.findall("snapshots-20[0-9]{2}-[0-9]{2}-[0-9]{2}$", subdir):
        photos.extend(os.listdir(os.path.join(photos_base_dir, subdir)))

photos = pd.DataFrame(photos, columns=["names"])

# extract time and cast to DateTime
photos["time"] = list(extract_time(i) for i in photos["names"])
photos["time"] = pd.to_datetime(photos["time"])

# sort DataFrames to apply pd.merge_asof
photos.sort_values(by="time", inplace=True)
observations.sort_values(by="Date_Time", inplace=True)

# merge nearest photo to observations
observations = pd.merge_asof(observations, photos,
                             left_on="Date_Time",
                             right_on="time",
                             direction="forward",
                             tolerance=tolerance
                             )

# drop rows with no photos
observations = observations.dropna(how="any", axis="index", subset=["names"])

# compute full paths
observations["full_path"] = list(get_full_path(i) for i in observations["names"])
photos["full_path"] = list(get_full_path(i) for i in photos["names"])

# reset index
observations = observations.reset_index(drop=True)

print("observations and photos was obtained")


def compute_correlations(row: pd.Series):
    first_image = get_masked_image(row["full_path"], masks)
    canal_names = "rgbhsv"

    result = {}
    result_is_empty = [True] * len(correlation_thresholds)

    def correlation(a, b):
        return np.corrcoef(a, b)[0, 1]

    photos_short = photos[(photos["time"] > (row["Date_Time"] + tolerance / 2)) &
                          (photos["time"] < (row["Date_Time"] + pd.Timedelta("1h")))]

    current_correlation = 1

    for k in range(len(photos_short)):
        if photos_short.iloc[k]["names"][-5] == row["names"][-5]:
            # store previous correlation to filter abnormality
            previous_correlation = current_correlation
            second_image = get_masked_image(photos_short.iloc[k]["full_path"], masks)

            # canal 5 because of V from HSV
            current_correlation = correlation(first_image[5], second_image[5])

            print(f"{k}##{current_correlation}", end="\r")

            # if there if a big gap in correlation, then pass this photo
            if abs(previous_correlation - current_correlation) > 0.25:
                # restore correlation
                current_correlation = previous_correlation
                continue

            for i, threshold in enumerate(correlation_thresholds):
                if current_correlation < threshold and result_is_empty[i]:
                    result_is_empty[i] = False
                    result["corr_on_threshold_" + str(threshold)] = current_correlation
                    result["photo_on_threshold_" + str(threshold)] = photos_short.iloc[k]["time"]
                    break

            if current_correlation < correlation_thresholds[0]:
                break
        a = 5

    return row.append(pd.Series(result))


observations = observations.apply(compute_correlations, axis=1)

observations.to_csv(os.path.join(databases_dir, "correlations"))

print(observations)
