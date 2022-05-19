import os
import re
import torch
import shutil
import pandas as pd
from regressor_on_resnet.nn_logging import Logger


class PretrainedLoader:
    def __init__(self):
        self.train = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.test = pd.DataFrame()

        self.model = None

    def init_using_paths(self, model_path, dataset_path, store_path=None):
        for piece in ('train', 'validation', 'test'):
            df = pd.read_csv(os.path.join(dataset_path, f'subset_{piece}.csv'), index_col=0)
            self.__setattr__(piece, df)

        self.model = torch.load(model_path)

        if store_path is not None:
            self.copy_data(model_path, dataset_path, store_path)

    def copy_data(self, model_path, dataset_path, store_path):
        files = zip(self._get_subset_paths(dataset_path) + [model_path],
                    self._get_subset_paths(store_path) +
                    [os.path.join(store_path, 'model_init.pt')])

        for source, destination in files:
            shutil.copyfile(source, destination)

    @staticmethod
    def _get_subset_paths(dataset_path):
        return list(map(lambda x: os.path.join(dataset_path, x),
                        (f'subset_{i}.csv' for i in ('train', 'validation', 'test'))))

    def init_using_logger(self, logger: Logger, base_run_number: int):
        misc_dirs = []
        tb_dirs = []
        for directory in os.listdir(logger.base_log_dir):
            if re.match(r'misc_\d{8}_\d{6}_' + str(base_run_number) + r'$', directory):
                misc_dirs.append(directory)
            if re.match(r'tb_\d{8}_\d{6}_' + str(base_run_number) + r'$', directory):
                tb_dirs.append(directory)

        assert len(misc_dirs) == 1, f'misc_dirs must be exactly 1, found{misc_dirs}'
        misc_dir = os.path.join(logger.base_log_dir, misc_dirs[0])

        # may be useful for appending new log data
        # assert len(tb_dirs) == 1, f'tb_dirs must be exactly 1, found{tb_dirs}'
        # tb_dir = os.path.join(logger.base_log_dir, tb_dirs[0])

        # examples for model names:
        #     model_ep0312.pt
        #     model_ep26.pt

        # get epoch's names as str
        epochs = tuple(map(
            lambda x: re.findall(r'\d+', x)[0],
            (i for i in os.listdir(misc_dir) if re.match(r'model_ep\d+\.pt', i))
        ))
        max_epoch = max(epochs, key=int)
        model_path = os.path.join(misc_dir, f'model_ep{max_epoch}.pt')

        self.init_using_paths(model_path, misc_dir, store_path=logger.misc_dir)
