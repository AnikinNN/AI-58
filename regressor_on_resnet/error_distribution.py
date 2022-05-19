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
from regressor_on_resnet.threadsafe_iterator import ThreadKiller, threaded_batches_feeder, threaded_cuda_feeder


logger = Logger(base_log_dir='/app/scripts/anikin/AI-58/regressor_on_resnet/logs')

base_run_number = 55

cuda_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
cpu_device = 'cpu'

batch_size = 64

pretrained_loader = PretrainedLoader()
pretrained_loader.init_using_logger(logger, base_run_number)

val_set = FluxDataset(flux_frame=pretrained_loader.validation,
                      batch_size=batch_size,
                      do_shuffle=True,
                      do_augment=False)

modified_resnet = pretrained_loader.model
modified_resnet.to(cuda_device)
modified_resnet.eval()

# start threads
cpu_queue_length = 3
cuda_queue_length = 3
preprocess_workers = [6, 15]

# contain train: [0] and validation: [1] queues
cpu_queues = [Queue(maxsize=cpu_queue_length), Queue(maxsize=cpu_queue_length)]
cuda_queues = [Queue(maxsize=cuda_queue_length), Queue(maxsize=cuda_queue_length)]
datasets = [None, val_set]

# one killer for all threads
threads_killer = ThreadKiller()
threads_killer.set_to_kill(False)

for i in [1]:
    for _ in range(1):
        cuda_thread = Thread(target=threaded_cuda_feeder,
                             args=(threads_killer,
                                   cuda_queues[i],
                                   cpu_queues[i],
                                   cuda_device))
        cuda_thread.start()
    for _ in range(preprocess_workers[i]):
        thr = Thread(target=threaded_batches_feeder, args=(threads_killer, cpu_queues[i], datasets[i]))
        thr.start()

result = np.zeros((0, 2))

for _ in trange(20000//batch_size):
    data_image, target, elevation = cuda_queues[1].get(block=True)
    data_out = modified_resnet(data_image, elevation)
    data_out = data_out.cpu().detach().numpy().reshape(batch_size)
    target = target.cpu().detach().numpy().reshape(batch_size)

    result = np.vstack((result, np.column_stack((data_out, target))))

print('absolute')
plt.scatter(result[:, 1], np.abs(result[:, 1] - result[:, 0]),)
# plt.hist2d(result[:, 1], np.abs(result[:, 1] - result[:, 0]), bins=400)
plt.savefig('absolute_error.png', dpi=150)
plt. clf()

print('relative')
plt.scatter(result[:, 1], np.abs((result[:, 1] - result[:, 0]) / result[:, 1]),)
# plt.hist2d(result[:, 1], np.abs((result[:, 1] - result[:, 0]) / result[:, 1]), bins=400)
plt.savefig('relative_error.png', dpi=150)
plt. clf()

print('vs')
plt.scatter(result[:, 1], result[:, 0],)
plt.plot((0, 1000), (0, 1000))
# plt.hist2d(result[:, 1], np.abs((result[:, 1] - result[:, 0]) / result[:, 1]), bins=400)
plt.savefig('versus.png', dpi=150)
plt. clf()

print('done')





























