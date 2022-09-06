import os
import sys

sys.path.append('/app/scripts/anikin/AI-58/')

import torch
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm, trange
from queue import Queue
from threading import Thread

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.train_common import train_model
from regressor_on_resnet.resnet_regressor import ResnetRegressor
from regressor_on_resnet.pretrained_loader import PretrainedLoader
from regressor_on_resnet.batch_factory import BatchFactory

logger = Logger(base_log_dir='/app/scripts/anikin/AI-58/regressor_on_resnet/logs')

base_run_number = 135

cuda_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
cpu_device = 'cpu'

batch_size = 32

pretrained_loader = PretrainedLoader()
pretrained_loader.init_using_logger(logger, base_run_number)
# pretrained_loader.init_using_paths('/app/scripts/anikin/AI-58/regressor_on_resnet/optuna_logs/misc_20220804_103529_1/model_ep54.pt',
#                                    './../logs/misc_20220526_125026_115')

val_set = FluxDataset(flux_frame=pretrained_loader.validation,
                      batch_size=batch_size,
                      do_shuffle=True)

val_factory = BatchFactory(val_set, cuda_device, False, to_variable=False)

modified_resnet = pretrained_loader.model
modified_resnet.to(cuda_device)
modified_resnet.eval()


result = np.zeros((0, 2))

for _ in trange(50000//batch_size):
    batch = val_factory.cuda_queue.get(block=True)
    data_out = modified_resnet(batch.images, batch.elevations)
    data_out = data_out.cpu().detach().numpy().reshape(batch_size)
    target = batch.fluxes.cpu().detach().numpy().reshape(batch_size)

    result = np.vstack((result, np.column_stack((data_out, target))))

np.save(f'evaluation_result_{base_run_number}.npy', result)
val_factory.stop()
print('done')





























