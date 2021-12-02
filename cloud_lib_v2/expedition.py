import json
import os.path
import re
import sys

import imageio
import swifter
import numpy as np
import pandas as pd
import multiprocessing as mp
from numpy import ma
from tqdm import tqdm

from cloud_lib_v2 import photos_processing, read_radiation, image_processing
from cloud_lib_v2.mask import Mask


class Expedition:
    """
        Class describes a singe expedition
        event oriented class
        event is a taken photo
    """

    def __init__(self, config_json_path):
        # cores
        self.core_number = 6
        # masks:
        #   key is a camera id
        #   value is an instance of Mask
        self.masks = dict()
        # determines usage of downscaling to improve time of calculations
        self.resize = None
        # probably we will need that
        self.expedition_name = None
        # path to radiation data
        # must contain only "CR20[0-9]{6}.txt" files
        self.radiation_dir = None
        # path to observations table
        self.observations_table = None
        # path to photos
        self.photos_base_dir = None
        # time tolerance to apply merge
        self.time_tolerance = None
        # autocorrelations below this threshold marks as anomalies
        self.anomaly_threshold = None

        # read config
        self.set_configuration_using_json(config_json_path)
        # make output dir
        self.output_dir = sys.argv[0][:-3] + "_output"
        try:
            os.mkdir(self.output_dir)
        except OSError:
            pass

        # dataFrames based on radiation measurement, photos and observations respectively
        self.df_radiation = pd.DataFrame()
        self.df_events = pd.DataFrame()
        self.df_observations = pd.DataFrame()

        # progress bar
        self.progress_bar = None

    def set_configuration_using_json(self, config_json_path):
        # load config from json
        with open(config_json_path) as config_json_file:
            config = json.load(config_json_file)

        # set resize flag
        self.resize = config["resize"]

        # load masks
        for file in os.listdir(config["mask_dir"]):
            # check that mask matches "mask_ID[0-9]+\.png"
            file = os.path.join(config["mask_dir"], file)
            if re.findall(r"mask_ID[0-9]+\.png$", file):
                mask = Mask(file, self.resize)
                self.masks[mask.camera_id] = mask

        # save constants
        self.expedition_name = config["expedition_name"]
        self.radiation_dir = config["radiation_dir"]
        self.observations_table = config["observations_table"]
        self.photos_base_dir = config["photos_base_dir"]
        self.time_tolerance = pd.Timedelta(config["time_tolerance"])
        self.anomaly_threshold = config["anomaly_threshold"]

    def init_events(self):
        self.df_events = photos_processing.init_events(self.photos_base_dir)

    def init_radiation(self):
        self.df_radiation = read_radiation.read_radiation_from_dir(self.radiation_dir)

    def init_observation(self):
        if self.observations_table.endswith("xlsx"):
            self.df_observations = pd.read_excel(self.observations_table)
        elif self.observations_table.endswith("csv"):
            self.df_observations = pd.read_csv(self.observations_table)

        self.df_observations.rename(columns={"Date_Time": "observation_datetime"}, inplace=True)
        self.df_observations["observation_datetime"] = pd.to_datetime(self.df_observations["observation_datetime"])
        self.df_observations.sort_values(by="observation_datetime", inplace=True)

    def sort_events(self):
        self.df_events.sort_values(by="photo_datetime", inplace=True)

    def merge_radiation_to_events(self, inplace=True):
        """
        merge DataFrames, for each df_events row find nearest row in time from df_radiation
        nearest means that we take nearest after photo, because of experiment design
        write everything in one DataFrame
        if there is no row in df_radiation that fits to tolerance condition, writes NaN

        inplace: write result of merge to df_events

        return: result of merge
        """
        self.sort_events()
        self.df_radiation.sort_values(by="radiation_datetime", inplace=True)

        merged = pd.merge_asof(self.df_events, self.df_radiation,
                               left_on="photo_datetime",
                               right_on="radiation_datetime",
                               direction="forward",
                               tolerance=self.time_tolerance
                               )
        if inplace:
            self.df_events = merged
        return merged

    def merge_observations_to_events(self, inplace=True):
        """
        merge DataFrames, for each df_events row find nearest row in time from df_observations
        nearest means that we take nearest after photo, because of experiment design
        write everything in one DataFrame
        if there is no row in df_observations that fits to tolerance condition, writes NaN

        inplace: write result of merge to df_events

        return: result of merge
        """
        self.sort_events()
        self.df_observations.sort_values(by="observation_datetime", inplace=True)

        merged = pd.merge_asof(self.df_events, self.df_observations,
                               left_on="photo_datetime",
                               right_on="observation_datetime",
                               direction="forward",
                               tolerance=self.time_tolerance
                               )
        if inplace:
            self.df_events = merged
        return merged

    def delete_outside_datetime(self, a_datetime, b_datetime):
        """
        deletes rows from df_observations, df_radiation, df_events
        that are out of range [a_datetime, b_datetime]
        """
        a_datetime = pd.to_datetime(a_datetime)
        b_datetime = pd.to_datetime(b_datetime)

        # swap if needed
        if a_datetime > b_datetime:
            a_datetime, b_datetime = b_datetime, a_datetime

        selection = (self.df_events["photo_datetime"] > a_datetime) & (
                self.df_events["photo_datetime"] < b_datetime)
        self.df_events = self.df_events[selection]

        selection = (self.df_radiation["radiation_datetime"] > a_datetime) & (
                self.df_radiation["radiation_datetime"] < b_datetime)
        self.df_radiation = self.df_radiation[selection]

        selection = (self.df_observations["observation_datetime"] > a_datetime) & (
                self.df_observations["observation_datetime"] < b_datetime)
        self.df_observations = self.df_observations[selection]

    def get_image(self, row: pd.Series):
        camera_id = row["camera_id"]
        img = imageio.imread(row["photo_path"])
        if self.resize:
            img = image_processing.resize4x(img).astype(np.uint8)

        if camera_id in self.masks.keys():
            img = np.reshape(ma.array(img, mask=self.masks[camera_id].mask), (-1, 3))

        canals = list(img[:, j] for j in range(3))
        canals.extend(image_processing.HSV(img))
        return canals

    def split_to_batches(self):
        pass

    def compute_statistic_features(self):
        print(f"computing_statistic_features")
        self.progress_bar = tqdm(total=self.df_events.shape[0], position=0, leave=True)
        self.df_events = self.df_events.apply(self._compute_statistic_features_for_one_event, axis=1)
        # with mp.Pool(self.core_number) as pool:
        #     result = pool.map(self._compute_statistic_features_for_one_event, self.df_events.iterrows())
        # self.df_events = self.df_events.apply(self._compute_statistic_features_for_one_event, axis=1)
        self.progress_bar.close()
        self.progress_bar = None

    def _compute_statistic_features_for_one_event(self, row: pd.Series):
        img = self.get_image(row)

        features = image_processing.calculate_features(img)
        features_index = list("feature" + str(i) for i in range(features.shape[0]))
        features_series = pd.Series(features, index=features_index)
        row = row.append(features_series)
        self.progress_bar.update()
        return row

    def _compute_correlation_to_previous_for_one_event(self, row: pd.Series):
        # find previous event
        prev_img = None
        prev_index = row.name
        while prev_img is None and prev_index > 0:
            prev_index -= 1
            if self.df_events.iloc[prev_index]["camera_id"] == row["camera_id"]:
                prev_img = self.get_image(self.df_events.iloc[prev_index])

        current_img = self.get_image(row)
        corr = image_processing.compute_correlation(current_img[5], prev_img[5]) if prev_img is not None else None
        self.progress_bar.update()
        return corr

    def compute_correlation_to_previous(self):

        self.sort_events()
        self.df_events = self.df_events.reset_index(drop=True)

        print("computing_correlation_to_previous")
        self.progress_bar = tqdm(total=self.df_events.shape[0], position=0, leave=True)
        self.df_events["correlation_to_previous"] = self.df_events.apply(
            self._compute_correlation_to_previous_for_one_event, axis=1)
        self.progress_bar.close()
        self.progress_bar = None
