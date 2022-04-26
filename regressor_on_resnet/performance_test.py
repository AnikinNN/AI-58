import os
import sys
import cProfile
sys.path.append(os.path.join(sys.path[0], '..'))

import re
import threading
from threading import Thread
from queue import Queue, Empty

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import imgaug.augmenters as iaa
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# from cloud_lib_v2.expedition import Expedition
# from cloud_lib_v2.mask import Mask
from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.metadata_loader import MetadataLoader
from regressor_on_resnet.threadsafe_iterator import ThreadKiller, threaded_batches_feeder, threaded_cuda_feeder


def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          cuda_batches_queue: Queue,
                          per_step_epoch: int,
                          current_epoch: int):
    model.eval()

    loss_values = []

    pbar = tqdm(total=per_step_epoch)
    pbar.set_description(desc='validation')
    for batch_idx in range(per_step_epoch):
        data_image, target = cuda_batches_queue.get(block=True)
        data_out = model(data_image)

        # if batch_idx == 0:
        #     plot_examples_segmentation(data_image, data_out, file_output = os.path.join(logs_basepath,
        #     'val_results_ep%04d.png' % current_epoch))

        loss = loss_function(data_out, target)
        loss_values.append(loss.item())
        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    return np.mean(loss_values)


def threaded_queue_cleaner(queue: Queue, killer: ThreadKiller):
    for _ in trange(10):
        queue.get(block=True)
    killer.to_kill = True

def train_model(model: torch.nn.Module,
                train_dataset,
                val_dataset,
                max_epochs=480):
    loss_function = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=32, T_mult=2, eta_min=1e-8)

    # start threads
    cpu_queue_length = 4
    cuda_queue_length = 4
    preprocess_workers = [0, 5]

    # contain train: [0] and validation: [1] queues
    cpu_queues = [Queue(maxsize=cpu_queue_length), Queue(maxsize=cpu_queue_length)]
    cuda_queues = [Queue(maxsize=cuda_queue_length), Queue(maxsize=cuda_queue_length)]
    datasets = [train_dataset, val_dataset]

    # one killer for all threads
    threads_killer = ThreadKiller()
    threads_killer.set_to_kill(False)

    for i in range(1, 2):
        # for _ in range(1):
        #     cuda_thread = Thread(target=threaded_cuda_feeder,
        #                          args=(threads_killer,
        #                                cuda_queues[i],
        #                                cpu_queues[i],
        #                                cuda_device))
        #     cuda_thread.start()
        for _ in range(preprocess_workers[i]):
            thr = Thread(target=threaded_batches_feeder, args=(threads_killer, cpu_queues[i], datasets[i]))
            thr.start()
        thr = Thread(target=threaded_queue_cleaner, args=(cpu_queues[i], threads_killer))
        thr.start()
        thr.join()

    return
    steps_per_epoch_train = 64
    steps_per_epoch_valid = len(val_dataset) // batch_size + 1

    for epoch in range(max_epochs):
        val_loss = validate_single_epoch(model,
                                         loss_function,
                                         cuda_queues[1],
                                         steps_per_epoch_valid,
                                         current_epoch=epoch)

    threads_killer.set_to_kill(True)

    # clean queues
    for queue_list in [cpu_queues, cuda_queues]:
        for i in queue_list:
            i.queue.clear()


cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = 'cpu'

metadata_loader = MetadataLoader('./AI-58-config.json',
                                 radiation_threshold=10,
                                 split=(0.6, 0.2, 0.2),
                                 store_path=None)

batch_size = 64
trainset = FluxDataset(flux_frame=metadata_loader.train,
                       batch_size=batch_size,
                       do_shuffle=True,
                       do_augment=True)

valset = FluxDataset(flux_frame=metadata_loader.validation,
                     batch_size=batch_size,
                     do_shuffle=False,
                     do_augment=False)

resnet50 = models.resnet50(pretrained=True, progress=False)

for param in resnet50.parameters():
    param.requires_grad = False

resnet50.fc = torch.nn.Identity()

modified_resnet = torch.nn.Sequential(resnet50,
                                      nn.Linear(in_features=2048, out_features=512),
                                      nn.ReLU(),
                                      nn.Linear(in_features=512, out_features=128),
                                      nn.ReLU(),
                                      nn.Linear(in_features=128, out_features=1),
                                      nn.ReLU())

modified_resnet.to(cuda_device)

train_model(modified_resnet,
            train_dataset=trainset,
            val_dataset=valset,
            max_epochs=480)

print()
