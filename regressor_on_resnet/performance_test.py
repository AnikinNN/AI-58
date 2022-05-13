import os
import sys
import cProfile

sys.path.append(os.path.join(sys.path[0], '..'))

from threading import Thread, active_count
from queue import Queue, Empty

import numpy as np
from tqdm import tqdm, trange
import torch

from regressor_on_resnet.flux_dataset_versions.flux_dataset_best import FluxDataset as fd_best
from regressor_on_resnet.flux_dataset_versions.flux_dataset_init import FluxDataset as fd_init
from regressor_on_resnet.flux_dataset_versions.flux_dataset_cv2 import FluxDataset as fd_cv2
from regressor_on_resnet.flux_dataset_versions.flux_dataset_concat import FluxDataset as fd_concat
from regressor_on_resnet.flux_dataset_versions.flux_dataset_mask import FluxDataset as fd_mask
from regressor_on_resnet.flux_dataset import FluxDataset as fd_current


from regressor_on_resnet.metadata_loader import MetadataLoader
from regressor_on_resnet.threadsafe_iterator import ThreadKiller, threaded_batches_feeder


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


def test_performance(train_dataset,
                     val_dataset,
                     preprocess_workers_):
    # start threads
    cpu_queue_length = 4
    cuda_queue_length = 4
    preprocess_workers = [preprocess_workers_, preprocess_workers_]

    # contain train: [0] and validation: [1] queues
    cpu_queues = [Queue(maxsize=cpu_queue_length), Queue(maxsize=cpu_queue_length)]
    cuda_queues = [Queue(maxsize=cuda_queue_length), Queue(maxsize=cuda_queue_length)]
    datasets = [train_dataset, val_dataset]

    # one killer for all threads
    threads_killer = ThreadKiller()
    threads_killer.set_to_kill(False)

    workers = []

    for i in [0]:
        for _ in range(preprocess_workers[i]):
            thr = Thread(target=threaded_batches_feeder, args=(threads_killer, cpu_queues[i], datasets[i]))
            thr.start()
            workers.append(thr)
        thr = Thread(target=threaded_queue_cleaner, args=(cpu_queues[i], threads_killer))
        thr.start()
        thr.join()

    threads_killer.set_to_kill(True)

    # clean queues
    while active_count() > 1:
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
for FluxDataset, name in [
    # [fd_init, 'fd_init'],
    # [fd_cv2, 'fd_cv2'],
    # [fd_concat, 'fd_concat'],
    # [fd_mask, 'fd_mask'],
    # [fd_best, 'fd_best'],
    [fd_current, 'fd_current'],
]:
    for proc_worker_num in list(range(1, 9)) + [10, 15, 20]:
    # for proc_worker_num in [5]:
        train_set = FluxDataset(flux_frame=metadata_loader.train,
                                batch_size=batch_size,
                                do_shuffle=True,
                                do_augment=True)

        val_set = FluxDataset(flux_frame=metadata_loader.validation,
                              batch_size=batch_size,
                              do_shuffle=False,
                              do_augment=False)

        print(f'{name}, {proc_worker_num}')
        test_performance(train_dataset=train_set,
                         val_dataset=val_set,
                         preprocess_workers_=proc_worker_num)

print()
