import argparse
import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import optuna

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.resnet_regressor import ResnetRegressor
from regressor_on_resnet.train_common import train_model
from regressor_on_resnet.pretrained_loader import PretrainedLoader

lr_limit = [1e-6, 1e-3]
fully_connected_depth_limit = [1, 6]
fully_connected_width_limit = [16, 1024]
batch_size_limit = [4, 32]

# set const parameters
dataset_path = './logs/misc_20220526_125026_115'
epochs = 60
events_on_training = 6_000_000

# select
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int, help='which cuda device use', dest='cuda_device_number', default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_number)
if torch.cuda.is_available():
    cuda_device = torch.device(f'cuda:0')
else:
    raise ValueError(f'cuda devise {args.cuda_device_number} is not available')


def objective(trial: optuna.trial.Trial):
    # sample some parameters
    learning_rate = trial.suggest_float('learning_rate', *lr_limit, log=True)
    fully_connected_depth = trial.suggest_int('fully_connected_depth', *fully_connected_depth_limit)
    fully_connected_widths = tuple(trial.suggest_int(f'fully_connected_width_{i}', *fully_connected_width_limit)
                                   for i in range(fully_connected_depth))
    batch_size = trial.suggest_int('batch_size', *batch_size_limit)

    steps_per_epoch_train = events_on_training // batch_size // epochs
    steps_per_epoch_valid = int(steps_per_epoch_train / 1.5)

    logger = Logger(base_log_dir='./optuna_logs')

    pretrained_loader = PretrainedLoader()
    pretrained_loader.load_dataset(dataset_path)

    train_set = FluxDataset(flux_frame=pretrained_loader.train,
                            batch_size=batch_size,
                            do_shuffle=True)

    val_set = FluxDataset(flux_frame=pretrained_loader.validation,
                          batch_size=batch_size,
                          do_shuffle=True)

    hard_mining_train_set = FluxDataset(flux_frame=pretrained_loader.train,
                                        batch_size=batch_size,
                                        do_shuffle=False)

    modified_resnet = ResnetRegressor(fully_connected_widths)
    modified_resnet.to(cuda_device)

    return train_model(modified_resnet,
                       train_dataset=train_set,
                       val_dataset=val_set,
                       hard_mining_dataset=hard_mining_train_set,
                       logger=logger,
                       learning_rate=learning_rate,
                       static_learning_rate=True,
                       cuda_device=cuda_device,
                       max_epochs=epochs,
                       steps_per_epoch_train=steps_per_epoch_train,
                       steps_per_epoch_valid=steps_per_epoch_valid,
                       train_convolutional_since_epoch=epochs // 2)


study = optuna.create_study(storage='sqlite:///optuna_logs/resnet_regressor_tuning.db',
                            study_name='resnet_regressor_tuning', load_if_exists=True, )
study.optimize(objective, n_trials=100)
