import datetime
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

# plt.switch_backend('agg')


class Logger:
    experiment = None

    def __init__(self, model_save_type: str, base_log_dir=None,):
        if base_log_dir is None:
            base_log_dir = os.path.join(os.path.dirname(sys.argv[0]), 'logs')
        self.base_log_dir = base_log_dir
        make_dir(self.base_log_dir)

        self.experiment_number = self.get_experiment_number()
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_dir = os.path.join(self.base_log_dir, f'tb_{datetime_str}_{self.experiment_number}')
        self.misc_dir = os.path.join(self.base_log_dir, f'misc_{datetime_str}_{self.experiment_number}')

        for i in [self.tb_dir, self.misc_dir]:
            make_dir(i)

        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)
        self.best_model_saver = BestModelSaver(model_save_type)

    def get_tb_writer(self):
        return self.tb_writer

    def get_experiment_number(self):
        numbers = set()
        for directory in os.listdir(self.base_log_dir):
            if re.match(r'(tb|misc)_\d{8}_\d{6}_\d+$', directory):
                numbers.add(int(directory.split('_')[-1]))
        return max(numbers) + 1 if len(numbers) else 1

    def store_batch_as_image(self, tag, batch: torch.Tensor, global_step=None, inv_normalizer=None):
        batch = batch.cpu().detach()
        if inv_normalizer is not None:
            batch = torch.stack([inv_normalizer(torch.squeeze(i)) for i in torch.split(batch, 1)], dim=0)
        self.tb_writer.add_images(tag, batch, global_step=global_step)

    def store_scatter_hard_mining_weights(self, hard_mining_frame, epoch):
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot()
        x = hard_mining_frame['CM3up[W/m2]'].to_numpy()
        y = hard_mining_frame['hard_mining_weight'].to_numpy()
        ax.grid()
        ax.scatter(x, y, s=1)
        self.tb_writer.add_figure('hard_mining_weights', [fig], epoch)

    def save_model(self, model: torch.nn.Module, val_metric: float, epoch: int):
        self.best_model_saver.save(model, val_metric, self, epoch)


class BestModelSaver:
    def __init__(self, saving_type: str):
        self.saving_type = saving_type
        self.best_val_epoch = -1

        if self.saving_type == 'max':
            self.best_val_metric = - float('Inf')
            self.compare = self.max_compare
        elif self.saving_type == 'min':
            self.best_val_metric = float('Inf')
            self.compare = self.min_compare
        else:
            raise ValueError(f'{saving_type=} but must be max or min')

    def save(self, model: torch.nn.Module, val_metric: float, logger: Logger, epoch: int):
        if self.compare(val_metric):
            # save new model
            torch.save(model, os.path.join(logger.misc_dir, f'model_ep{epoch}.pt'))
            # delete old model
            old_model_path = os.path.join(logger.misc_dir, f'model_ep{self.best_val_epoch}.pt')
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
            # update best values
            self.best_val_metric = val_metric
            self.best_val_epoch = epoch

    def max_compare(self, new_value):
        return new_value > self.best_val_metric

    def min_compare(self, new_value):
        return new_value < self.best_val_metric


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
