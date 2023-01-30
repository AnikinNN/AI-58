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

        first_line = input_file.readline().strip()
        if first_line.count('.') > 0 and first_line.count(',') == 0:
            decimal = '.'
        elif first_line.count('.') == 0 and first_line.count(',') > 0:
            decimal = ','
        else:
            raise NotImplementedError(
                f'found {first_line.count(".")} dots and {first_line.count(",")}'
                f' commas at first line with numerical data of file {path}.'
                f' Don\'t know how to deal with it')

    # read data
    result = pd.read_csv(path, skiprows=14, sep="\t", header=None, decimal=decimal)
    # delete empty columns
    result = result.drop(labels=[5, 6, 7, 9], axis='columns')
    # assign column names
    result.columns = columns

    result["radiation_datetime"] = pd.to_datetime(result['data time'], format="%d/%m/%Y %H:%M:%S")

    corrupted_selection = result["radiation_datetime"].isna()
    if corrupted_selection.sum() > 0:
        print(f'{path}, has {corrupted_selection.sum()} mistakes:\n',
              result[corrupted_selection])

    return result


def read_radiation_from_dir(radiation_dir):
    # create dataframe for radiation data
    radiation = pd.DataFrame()

    for file in os.listdir(radiation_dir):
        # filter file names
        if re.findall(r"CR20\d{6}.txt", file):
            radiation = pd.concat((radiation, read_single_radiation_file(os.path.join(radiation_dir, file))),
                                  ignore_index=True)
    # radiation["radiation_datetime"] = pd.to_datetime(radiation['data time'], format="%d/%m/%Y %H:%M:%S")
    radiation.drop(columns='data time', inplace=True)
    radiation.sort_values(by="radiation_datetime", inplace=True)
    return radiation
