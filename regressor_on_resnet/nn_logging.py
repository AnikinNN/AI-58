import datetime
import os
import re
import sys

from torch.utils.tensorboard import SummaryWriter


class Logger:
    experiment = None

    def __init__(self, base_log_dir=None):
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

    def get_tb_writer(self):
        return self.tb_writer

    def get_experiment_number(self):
        numbers = set()
        for directory in os.listdir(self.base_log_dir):
            if re.match(r'(tb|misc)_\d{8}_\d{6}_\d+$', directory):
                numbers.add(int(directory.split('_')[-1]))
        return max(numbers) + 1 if len(numbers) else 1


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
