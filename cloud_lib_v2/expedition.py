import json
import os.path
import re
import sys

import pandas as pd

from cloud_lib_v2 import photos_processing, read_radiation
from cloud_lib_v2.mask import Mask


class Expedition:
    """
        Класс описывает отдельно взятую экспедицию
    """

    def __init__(self, config_json_path):
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
        # path to observations data
        self.observations_dir = None

        self.photos_base_dir = None
        self.time_tolerance = None
        self.anomaly_threshold = None

        # read config
        self.set_configuration_using_json(config_json_path)
        # make output dir
        self.output_dir = sys.argv[0][:-3] + "_output"
        try:
            os.mkdir(self.output_dir)
        except OSError as error:
            pass

        # dataFrames based on radiation measurement and photos respectively
        self.df_radiation = pd.DataFrame()
        self.df_events = pd.DataFrame()
        print()

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
        self.observations_dir = config["observations_dir"]
        self.photos_base_dir = config["photos_base_dir"]
        self.time_tolerance = pd.Timedelta(config["time_tolerance"])
        self.anomaly_threshold = config["anomaly_threshold"]

    def init_events(self):
        """
            searches all photos in self.photos_base_dir
            creates new DataFrame with photos
            fields columns:
                photo_name
                photo_path
                photo_datetime
                camera_id
            sorts by photo_datetime
        """
        photo_names = photos_processing.get_photo_names(self.photos_base_dir)
        self.df_events = pd.DataFrame(photo_names, columns=["photo_name"])

        self.df_events["photo_path"] = self.df_events.apply(
            lambda x: photos_processing.get_full_path(x["photo_name"], self.photos_base_dir),
            axis=1
        )

        self.df_events["camera_id"] = self.df_events.apply(
            lambda x: int(x["photo_name"][28: -4]),
            axis=1)

        self.df_events["photo_datetime"] = self.df_events.apply(
            lambda x: photos_processing.extract_time(x["photo_name"]),
            axis=1)
        self.df_events["photo_datetime"] = pd.to_datetime(self.df_events["photo_datetime"])

        self.df_events.sort_values(by="photo_datetime", inplace=True)

    def init_radiation(self):
        self.df_radiation = read_radiation.read_radiation_from_dir(self.radiation_dir)

    def init_observation(self):
        pass
