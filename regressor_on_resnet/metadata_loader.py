import os.path

import pandas as pd
from sklearn.model_selection import train_test_split

from cloud_lib_v2.expedition import Expedition


class MetadataLoader:
    def __init__(self, configs, radiation_threshold=10, split=(0.80, 0.10, 0.10), store_path=None):
        self.all_df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.test = pd.DataFrame()

        self.radiation_threshold = radiation_threshold

        # now this implemented using cloud_lib_v2
        for config in configs:
            self.load_data(config)

        self.all_df.sort_values(by="photo_datetime", inplace=True)
        self.all_df['hard_mining_weight'] = 1.0
        # implement classes
        self.split(*split)
        if store_path is not None:
            self.store_splits(store_path)

    def load_data(self, config):
        expedition = Expedition()
        if isinstance(config, dict):
            expedition.init_using_config_dict(config)
        else:
            expedition.init_using_json(config)
        expedition.init_events()
        expedition.init_radiation()
        expedition.init_elevation()

        expedition.merge_radiation_to_events()
        expedition.merge_elevation_to_events()
        df = expedition.df_events[expedition.df_events['CM3up[W/m2]'] > self.radiation_threshold]
        df = df.dropna(subset=['sun_altitude'])
        df['expedition_name'] = expedition.expedition_name
        df.reset_index(drop=True, inplace=True)
        self.extend_all(df)

    def extend_all(self, appendix):
        self.all_df = pd.concat((self.all_df, appendix), axis=0, ignore_index=True)

    def split(self, train_size, validation_size, test_size):
        assert (train_size + validation_size + test_size) <= 1, 'sum of train, validation, test must be less than 1'
        self.all_df['date_hour'] = pd.to_datetime(self.all_df['photo_datetime'].dt.date) + \
                                   pd.to_timedelta(self.all_df['photo_datetime'].dt.hour, unit='hours')

        train, test = train_test_split(self.all_df['date_hour'].unique(),
                                       test_size=test_size,
                                       train_size=train_size + validation_size)
        train, validation = train_test_split(train,
                                             test_size=validation_size / (train_size + validation_size),
                                             train_size=train_size / (train_size + validation_size))

        self.train = self.all_df[self.all_df['date_hour'].isin(train)]
        self.validation = self.all_df[self.all_df['date_hour'].isin(validation)]
        self.test = self.all_df[self.all_df['date_hour'].isin(test)]

        for df, name in [(self.all_df, 'overall'),
                         (self.train, 'train'),
                         (self.validation, 'validation'),
                         (self.test, 'test')]:
            print(f'{name} len: {df.shape[0]}')

    def store_splits(self, path):
        for df, name in [(self.train, 'train'),
                         (self.validation, 'validation'),
                         (self.test, 'test')]:
            df.to_csv(os.path.join(path, f'subset_{name}.csv'))
