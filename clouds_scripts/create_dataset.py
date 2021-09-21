import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
from compute_features import calculate_masked_features, features_calculator
from connect_rad_photos import connect_rad_photos, extract_time, get_full_path
from multiprocessing import Pool, freeze_support
from path import *


def calc(x):
    x = os.path.join(photos_base_dir, "snapshots-" + extract_time(x)[:10], x)
    return calculate_masked_features(x, masks)


def to_batches(items):
    """
    split items to some groups by process_number elements
    :param items:
    :return: list of list of items
    """
    result = []
    batch_size = len(items) // process_number

    for i in range(batch_size):
        result.append(items[i * process_number: (i + 1) * process_number])
    if batch_size * process_number < len(items):
        result.append(items[batch_size * process_number:])

    return result


if __name__ == '__main__':
    freeze_support()

    if len(sys.argv) > 1:
        photos_dirs = [sys.argv[1]]

    for photos_dir in photos_dirs:
        # get radiations and photos
        dataset = connect_rad_photos(photos_dir,
                                     rad_dir,
                                     tolerance,
                                     save_to_path=os.path.join(databases_dir, "photos_to_rad.csv")
                                     )

        # delete unused photos or filter them
        start_date = pd.to_datetime("2021-08-02 00:00:00")
        end_date = pd.to_datetime("2021-08-02 00:2:00")
        selection = (dataset["photo_time"] > start_date) & (dataset["photo_time"] < end_date)
        # apply selection
        dataset = dataset[selection]

        print("\nFiltered dataset:")
        print(dataset)

        photo_names = list(get_full_path(i) for i in dataset["photos"])
        batches = to_batches(photo_names)

        calculator = features_calculator(masks)

        features = np.zeros((0, 162))
        timer = datetime.now()

        with Pool(processes=process_number) as p:
            for batch in tqdm(batches, total=len(batches), ncols=100):
                features = np.vstack((features, p.map(calculator, batch)))
        timer = datetime.now() - timer
        features = np.vstack(features)

        # code below for single core, not tested
        if 0:
            def get_features(row: pd.Series):
                """
                function to use in dataset.apply()
                computes features and appends them to a series
                :param row:
                :return: series to replace with
                """
                path = os.path.join(photos_base_dir, row["photos"])

                # print something to let us know that process is not freeze
                print(path, end="\r")
                return row.append(pd.Series(features))


            print("This photo was calculated:")
            timer = datetime.now()
            dataset = dataset.apply(get_features, axis=1)
            timer = datetime.now() - timer
            print("\nfinished")

        # reset index for fine joining
        dataset = dataset.reset_index(drop=True)
        # set columns names explicit as strings
        features = pd.DataFrame(features, columns=["f" + str(i) for i in range(features.shape[1])])
        # join dataset and features
        dataset = dataset.join(features, how="outer")

        # some logs
        print("\nDataset with features:")
        print(dataset)
        print("\nRunning time:")
        print(timer)
        print("Time per photo:")
        print((timer / dataset.shape[0]))

        # save dataset
        # if there is some calculated data in file, only recently calculated data should be overwritten or appended
        old_dataset = pd.DataFrame()

        dataset_path = os.path.join(databases_dir, "dataset.csv")

        if os.path.exists(dataset_path):
            old_dataset = pd.read_csv(dataset_path)
            old_dataset = pd.concat([old_dataset, dataset])
            old_dataset = old_dataset.drop_duplicates(subset="photos", keep="last")
        else:
            old_dataset = dataset
        old_dataset.to_csv(dataset_path, index=False)
