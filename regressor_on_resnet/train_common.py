import torch
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm
from queue import Queue
from threading import Thread

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.threadsafe_iterator import ThreadKiller, threaded_batches_feeder, threaded_cuda_feeder

import numpy as np
import os


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function: torch.nn.Module,
                       cuda_batches_queue: Queue,
                       per_step_epoch: int,
                       current_epoch: int,
                       logger: Logger):
    model.train()
    loss_values = []
    loss_tb = []
    pbar = tqdm(total=per_step_epoch, )
    pbar.set_description(desc='train')
    for batch_idx in range(per_step_epoch):
        data_image, target, elevation = cuda_batches_queue.get(block=True)

        if batch_idx == 0:
            logger.store_batch_as_image('train_batch', data_image,
                                        global_step=current_epoch,
                                        inv_normalizer=FluxDataset.inv_normalizer)

        target = Variable(target)

        optimizer.zero_grad()
        data_out = model(data_image, elevation)
        loss = loss_function(data_out, target)
        loss_values.append(loss.item())

        loss_tb.append(loss.item())
        logger.tb_writer.add_scalar('train_loss_per_step', np.mean(loss_tb), current_epoch * per_step_epoch + batch_idx)
        loss_tb = []

        loss.backward()
        optimizer.step()
        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    return np.mean(loss_values)


def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          cuda_batches_queue: Queue,
                          per_step_epoch: int,
                          current_epoch: int,
                          logger: Logger):
    model.eval()
    loss_values = []

    pbar = tqdm(total=per_step_epoch)
    pbar.set_description(desc='validation')
    for batch_idx in range(per_step_epoch):
        data_image, target, elevation = cuda_batches_queue.get(block=True)

        with torch.no_grad():
            data_out = model(data_image, elevation)

        if batch_idx == 0:
            logger.store_batch_as_image('val_batch', data_image,
                                        global_step=current_epoch,
                                        inv_normalizer=FluxDataset.inv_normalizer)

        loss = loss_function(data_out, target)
        loss_values.append(loss.item())
        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    return np.mean(loss_values)


def train_model(model: torch.nn.Module,
                train_dataset,
                val_dataset,
                logger: Logger,
                cuda_device,
                max_epochs=480):
    loss_function = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=32, T_mult=2, eta_min=1e-8)

    # start threads
    cpu_queue_length = 3
    cuda_queue_length = 3
    preprocess_workers = [6, 15]

    # contain train: [0] and validation: [1] queues
    cpu_queues = [Queue(maxsize=cpu_queue_length), Queue(maxsize=cpu_queue_length)]
    cuda_queues = [Queue(maxsize=cuda_queue_length), Queue(maxsize=cuda_queue_length)]
    datasets = [train_dataset, val_dataset]

    # one killer for all threads
    threads_killer = ThreadKiller()
    threads_killer.set_to_kill(False)

    for i in range(2):
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

    steps_per_epoch_train = 1024
    steps_per_epoch_valid = 512

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1} / {max_epochs}')
        train_loss = train_single_epoch(model,
                                        optimizer,
                                        loss_function,
                                        cuda_queues[0],
                                        steps_per_epoch_train,
                                        current_epoch=epoch,
                                        logger=logger)
        logger.tb_writer.add_scalar('train_loss_per_epoch', train_loss, epoch)

        val_loss = validate_single_epoch(model,
                                         loss_function,
                                         cuda_queues[1],
                                         steps_per_epoch_valid,
                                         current_epoch=epoch,
                                         logger=logger)
        logger.tb_writer.add_scalar('val_loss', val_loss, epoch)

        print(f'Validation loss: {val_loss}')

        lr_scheduler.step()

        torch.save(model, os.path.join(logger.misc_dir, f'model_ep{epoch}.pt'))

    threads_killer.set_to_kill(True)

    # clean queues
    # todo this is not good implementation copy from performance_test
    for queue_list in [cpu_queues, cuda_queues]:
        for i in queue_list:
            i.queue.clear()
