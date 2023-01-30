import os
import re
import sys

import pandas as pd


def get_photo_names(photos_path):
    photos = []
    # get list of photos names
    if os.path.isdir(photos_path):
        # there can be two situations:
        #     - photos_path contains photos: "img-2021-07-26T15-25-46devID1.jpg"
        #     - photos_path contains subdirectories: "snapshots-2021-07-26"
        photos_path_content = os.listdir(photos_path)
        for content in photos_path_content:
            if re.findall(r"img-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}devID\d+.jpg", content):
                photos.append(content)
            elif re.findall(r"snapshots-\d{4}-\d{2}-\d{2}\Z", content):
                photos.extend(os.listdir(os.path.join(photos_path, content)))
    return photos


def get_full_path(photo_name, photos_base_dir):
    raise NotImplementedError
    return os.path.join(photos_base_dir, "snapshots-" + extract_time(photo_name)[:10], photo_name)


def get_full_path_on_series(photo_names: pd.Series, photos_base_dir: str):
    if sys.platform.startswith("win"):
        separator = "\\"
    else:
        separator = "/"
    separators = "\\/"

    date = photo_names.map(lambda x: x[4:14])

    if photos_base_dir[-1] not in separators:
        photos_base_dir += separator

    return photos_base_dir + "snapshots-" + date + separator + photo_names


def extract_time(file_name):
    """
    extracts time and date from file_name
    :param file_name: string, example "img-2021-08-07T18-02-19devID1.jpg"
    :return: string, example "2021-08-07 18:02:19"
    """
    result = re.findall(r"\d{4}-\d{2}-\d{2}", file_name)[0]
    result += " "
    result += re.findall(r"[tT]\d{2}-\d{2}-\d{2}", file_name)[0][1:].replace("-", ":")
    return result


def init_events(photos_base_dir):
    """
        searches all photos in photos_base_dir
        creates new DataFrame with photos
        fields columns:
            photo_name
            photo_path
            photo_datetime
            camera_id
        sorts by photo_datetime
    """
    photo_names = get_photo_names(photos_base_dir)
    df_events = pd.DataFrame(photo_names, columns=["photo_name"])

    df_events["photo_path"] = get_full_path_on_series(df_events["photo_name"], photos_base_dir)

    df_events["camera_id"] = df_events["photo_name"].str.slice(28, -4)

    # img-2021-08-07T18-02-19devID1.jpg
    df_events["photo_datetime"] = pd.to_datetime(df_events["photo_name"].str.slice(4, 23), format='%Y-%m-%dT%H-%M-%S')

    df_events.sort_values(by="photo_datetime", inplace=True)

    return df_events
