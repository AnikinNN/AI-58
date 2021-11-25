import os
import re

import pandas as pd


def read_single_radiation_file(path):
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


def read_radiation_from_dir(radiation_dir):
    # create dataframe for radiation data
    radiation = pd.DataFrame()

    for file in os.listdir(radiation_dir):
        # filter file names
        if re.findall("CR20[0-9]{6}.txt", file):
            radiation = radiation.append(read_single_radiation_file(os.path.join(radiation_dir, file)))
    radiation["radiation_datetime"] = pd.to_datetime(radiation['data time'], format="%d/%m/%Y %H:%M:%S")
    radiation.drop(columns='data time', inplace=True)
    radiation.sort_values(by="radiation_datetime", inplace=True)
    return radiation
