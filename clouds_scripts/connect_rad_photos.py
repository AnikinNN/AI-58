import os
import re
import pandas as pd
from path import *


def get_full_path(photo_name):
    return os.path.join(photos_base_dir, "snapshots-" + extract_time(photo_name)[:10], photo_name)


def extract_time(file_name):
    """
    extracts time and date from file_name
    :param file_name: string, example "img-2021-08-07T18-02-19devID1.jpg"
    :return: string, example "2021-08-07 18:02:19"
    """
    result = re.findall("[0-9]{4}-[0-9]{2}-[0-9]{2}", file_name)[0]
    result += " "
    result += re.findall("T[0-9]{2}-[0-9]{2}-[0-9]{2}", file_name)[0][1:].replace("-", ":")
    return result


def connect_rad_photos(photos_path, rad_path, tolerance: pd.Timedelta, save_to_path=None):
    photos = []

    # get list of photos names
    if os.path.isdir(photos_path):
        photos.extend(os.listdir(photos_path))

    # create DataFrame
    photos = pd.DataFrame(photos, columns=["photos"])

    # extract time and cast to DateTime
    photos["photo_time"] = list(extract_time(i) for i in photos["photos"])
    photos["photo_time"] = pd.to_datetime(photos["photo_time"])

    def read_rad(path):
        with open(path) as input_file:
            # skip 13 lines
            for i in range(13):
                input_file.__next__()
            # get column names, delete empty column
            columns = []
            for item in input_file.readline().strip().split("\t"):
                if item != "none[V]":
                    columns.append(item)

        # read data
        result = pd.read_table(path, skiprows=14, delimiter="\t", header=None)
        # delete empty columns
        result = result.drop(labels=[5, 6, 7, 9], axis='columns')
        # assign column names
        result.columns = columns
        return result

    # create dataframe for radiation data
    radiation = pd.DataFrame()

    for file in os.listdir(rad_path):
        # filter file names
        if re.findall("CR20[0-9]{6}.txt", file):
            radiation = radiation.append(read_rad(os.path.join(rad_path, file)))
    radiation["data time"] = pd.to_datetime(radiation['data time'], format="%d/%m/%Y %H:%M:%S")

    # sort both dataframes
    photos.sort_values(by="photo_time", inplace=True)
    radiation.sort_values(by="data time", inplace=True)

    # merge DataFrames, for each photo row find nearest row in time from radiation DataFrame
    # nearest means that we take nearest after photo, because of experiment design
    # write everything in one DataFrame
    # if there is no row in radiation DataFrame that fits to tolerance condition, writes NaN, that need to be deleted
    merged = pd.merge_asof(photos, radiation,
                           left_on="photo_time",
                           right_on="data time",
                           direction="forward",
                           tolerance=tolerance
                           )
    # delete rows with NaN
    merged = merged.dropna()
    # save_to_path merged table
    if save_to_path:
        merged.to_csv(save_to_path)

    # set condition to False to disable printing
    if 0:
        print("connect_rad_photos() log:\n")
        print("photos DataFrame:")
        print(photos)
        print("\n")
        print("radiation DataFrame:")
        print(radiation)
        print("\n")
        print("merged DataFrame:")
        print(merged)
        print("\n")

    return merged


if __name__ == '__main__':
    """
        test
        probably there is another better way to do tests
    """
    from path import *

    synthetic_database_path = os.path.join(databases_dir, "photos_to_rad_synthetic.csv")

    photos_path = os.path.join(photos_base_dir, "snapshots-2021-08-03")

    connect_rad_photos(photos_path, rad_dir, tolerance,
                       save_to_path=synthetic_database_path)

    assert os.path.exists(synthetic_database_path)
    assert os.path.getsize(synthetic_database_path) > 100
    # there is a gap in radiation records
    assert len(pd.read_csv(synthetic_database_path)) == len(os.listdir(photos_path)) - 22

    photos_path = os.path.join(photos_base_dir, "snapshots-2021-07-26")
    connect_rad_photos(photos_path, rad_dir, tolerance,
                       save_to_path=synthetic_database_path)
    assert os.path.exists(synthetic_database_path)
    # if Dataframe is empty, but there is a header, that is why size is never zero
    assert os.path.getsize(synthetic_database_path) < 100
    assert len(pd.read_csv(synthetic_database_path)) == 0



