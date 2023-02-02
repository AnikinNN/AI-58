import json
import os.path
import re
import warnings
from pathlib import Path

import imageio
import pandas as pd
from numpy import ma
from tqdm import tqdm

from cloud_lib_v2 import photos_processing, read_radiation, image_processing
from cloud_lib_v2.observation import init_observation
from cloud_lib_v2.mask import Mask


class Expedition:
    """
        Class describes a singe expedition
        event oriented class
        event is a taken photo
    """

    def __init__(self):
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
        self.observation_table = None
        # path to elevation table
        self.elevation_table = None
        # path to photos
        self.photos_base_dir = None
        # time tolerance to apply merge
        self.time_tolerance = None
        # autocorrelations below this threshold marks as anomalies
        self.anomaly_threshold = None

        # dataFrames based on radiation measurement, photos and observations respectively
        self.df_radiation = pd.DataFrame()
        self.df_events = pd.DataFrame()
        self.df_observation = pd.DataFrame()
        self.df_elevation = pd.DataFrame()

        # progress bar
        self.progress_bar = None

    def init_using_json(self, config_json_path):
        # load config from json
        with open(config_json_path) as config_json_file:
            config = json.load(config_json_file)

        self.init_using_config_dict(config)

    def init_using_config_dict(self, config):
        for attribute in ("expedition_name",
                          "radiation_dir",
                          "observation_table",
                          "elevation_table",
                          "photos_base_dir",
                          "time_tolerance",
                          "anomaly_threshold",
                          "resize",
                          ):
            if attribute in config:
                setattr(self, attribute, config[attribute])

        self.time_tolerance = pd.to_timedelta(self.time_tolerance)

        if 'mask_dir' in config:
            self.load_masks(config["mask_dir"])

    def load_masks(self, mask_dir):
        # load masks
        for file in os.listdir(mask_dir):
            # check that mask matches "mask_ID[0-9]+\.png"
            if re.match(r"mask-id\d+\.png$", file):
                file = os.path.join(mask_dir, file)
                mask = Mask(file, self.resize)
                self.masks[mask.camera_id] = mask

    def init_events(self):
        if self.photos_base_dir is None:
            raise TypeError('self.photos_base_dir must be specified')
        self.df_events = photos_processing.init_events(self.photos_base_dir)

    def init_radiation(self):
        if self.radiation_dir is None:
            raise TypeError('self.radiation_dir must be specified')
        self.df_radiation = read_radiation.read_radiation_from_dir(self.radiation_dir)

    def init_observation(self):
        if self.observation_table is None:
            raise TypeError('self.observation_table must be specified')

        self.df_observation = init_observation(Path(self.observation_table))

    def init_elevation(self):
        if self.elevation_table is None:
            raise TypeError('self.elevation_table must be specified')
        self.df_elevation = pd.read_csv(self.elevation_table, sep=';')
        self.df_elevation['elevation_datetime'] = pd.to_datetime(self.df_elevation['datetime_UTC'],
                                                                 format='%Y-%m-%dT%H-%M-%S.%f')
        self.df_elevation.drop(columns=['datetime_UTC'], inplace=True)

    def sort_events(self):
        self.df_events.sort_values(by="photo_datetime", inplace=True)

    def merge_radiation_to_events(self, inplace=True):
        """
        merge DataFrames, for each df_events row find the nearest row in time from df_radiation
        the nearest means that we take nearest after photo, because of experiment design
        write everything in one DataFrame
        if there is no row in df_radiation that fits to tolerance condition, writes NaN

        inplace: write result of merge to df_events

        return: result of merge
        """
        return self._merge_something_to_events(target='radiation', inplace=inplace)

    def merge_observation_to_events(self, inplace=True):
        """
        merge DataFrames, for each df_events row find the nearest row in time from df_observation
        the nearest means that we take nearest after photo, because of experiment design
        write everything in one DataFrame
        if there is no row in df_observation that fits to tolerance condition, writes NaN

        inplace: write result of merge to df_events

        return: result of merge
        """
        return self._merge_something_to_events(target='observation', inplace=inplace)

    def merge_elevation_to_events(self, inplace=True):
        return self._merge_something_to_events(target='elevation', inplace=inplace)

    def _merge_something_to_events(self, target=None, inplace=True):
        targets = ["radiation", "observation", "elevation"]

        if not isinstance(self.time_tolerance, pd.Timedelta):
            raise TypeError(f'self.time_tolerance must be pd.Timedelta but got {type(self.time_tolerance)}')
        self.sort_events()
        if target in targets:
            datetime_column = f'{target}_datetime'
            attr = f'df_{target}'
        else:
            raise ValueError(f'target must be one of {targets}')

        df = self.__getattribute__(attr)
        if datetime_column not in df.columns:
            raise KeyError(f'{datetime_column} must be in self.{attr}.columns. Probably you forgot init that DataFrame')
        if not df.shape[0]:
            warnings.warn(f'self.{attr} has 0 rows')

        df.sort_values(by=datetime_column, inplace=True)
        merged = pd.merge_asof(self.df_events, df,
                               left_on="photo_datetime",
                               right_on=datetime_column,
                               direction="nearest",
                               tolerance=self.time_tolerance
                               )
        if inplace:
            self.df_events = merged
        return merged

    def delete_outside_datetime(self, a_datetime, b_datetime):
        """
        deletes rows from df_observation, df_radiation, df_events
        that are out of range [a_datetime, b_datetime]
        """
        a_datetime = pd.to_datetime(a_datetime)
        b_datetime = pd.to_datetime(b_datetime)

        # swap if needed
        if a_datetime > b_datetime:
            a_datetime, b_datetime = b_datetime, a_datetime

        for df_name, datetime_column in (('df_events', 'photo_datetime'),
                                         ('df_radiation', 'radiation_datetime'),
                                         ('df_observation', 'observation_datetime')):
            df = self.__getattribute__(df_name)
            if datetime_column in df.columns:
                selection = (df[datetime_column] > a_datetime) & \
                            (df[datetime_column] < b_datetime)
                setattr(self, df_name, df[selection])

    def get_image(self, row: pd.Series):
        camera_id = row["camera_id"]
        img = imageio.imread(row["photo_path"])
        if self.resize:
            img = image_processing.resize4x(img)

        if int(camera_id) in self.masks.keys():
            img = ma.array(img, mask=~self.masks[camera_id].mask)

        canals = list(img[:, :, j] for j in range(3))
        canals.extend(image_processing.to_hsv(img))
        return canals

    def split_to_batches(self):
        pass

    def compute_statistic_features(self):
        print(f"computing_statistic_features")
        self.progress_bar = tqdm(total=self.df_events.shape[0], position=0, leave=True)
        self.df_events = self.df_events.apply(self._compute_statistic_features_for_one_event, axis=1)
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
