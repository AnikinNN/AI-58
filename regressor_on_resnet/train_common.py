import pickle

import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import MSELoss

from tqdm import tqdm
from queue import Queue, Empty

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.sgdr_restarts_warmup import CosineAnnealingWarmupRestarts
from regressor_on_resnet.batch_factory import BatchFactory

import numpy as np
import os

from regressor_on_resnet.weighted_mse import WeightedMse, get_weights_and_bounds


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function,
                       cuda_batches_queue: Queue,
                       per_step_epoch: int,
                       current_epoch: int,
                       logger: Logger,
                       lr_scheduler
                       ):
    model.train()
    loss_values = []
    loss_tb = []
    pbar = tqdm(total=per_step_epoch, )
    pbar.set_description(desc='train')
    for batch_idx in range(per_step_epoch):
        data_image, target, elevation, _, hard_mining_weights = cuda_batches_queue.get(block=True)

        if batch_idx == 0:
            logger.store_batch_as_image('train_batch', data_image,
                                        global_step=current_epoch,
                                        inv_normalizer=FluxDataset.inv_normalizer)

        target = Variable(target)

        optimizer.zero_grad()
        data_out = model(data_image, elevation)
        loss = loss_function(data_out, target, hard_mining_weights)
        loss_values.append(loss.item())

        loss_tb.append(loss.item())
        logger.tb_writer.add_scalar('train_loss_per_step', np.mean(loss_tb), current_epoch * per_step_epoch + batch_idx)
        loss_tb = []

        loss.backward()
        optimizer.step()
        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})
        lr_scheduler.step()

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
        data_image, target, elevation, _, _ = cuda_batches_queue.get(block=True)

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


def calculate_hard_mining_weights(model: torch.nn.Module,
                                  train_dataset: FluxDataset,
                                  hard_mining_dataset: FluxDataset,
                                  cuda_batches_queue: Queue,
                                  logger: Logger,
                                  epoch: int
                                  ):
    model.eval()

    batch_number = len(hard_mining_dataset) // hard_mining_dataset.batch_size + 1
    pbar = tqdm(total=batch_number)
    pbar.set_description(desc='hard_mining')
    for batch_idx in range(batch_number):
        data_image, target, elevation, row_ids, _ = cuda_batches_queue.get(block=True)

        with torch.no_grad():
            data_out = model(data_image, elevation)
            error = torch.abs(target - data_out).cpu().detach().numpy().flatten()

        error = error / error.mean()
        updater_df = pd.DataFrame({'hard_mining_weight': error.tolist()})
        updater_df.set_index(pd.Index(row_ids), inplace=True)
        train_dataset.flux_frame.update(updater_df)

        logger.store_scatter_hard_mining_weights(train_dataset.flux_frame, epoch)

        pbar.update()
        pbar.set_postfix({'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    return


def train_model(model: torch.nn.Module,
                train_dataset: FluxDataset,
                val_dataset: FluxDataset,
                hard_mining_dataset: FluxDataset,
                logger: Logger,
                cuda_device,
                max_epochs=480,
                use_warmup: bool = False):
    steps_per_epoch_train = 1536
    steps_per_epoch_valid = 1024

    mse = MSELoss()
    weights, bounds = get_weights_and_bounds(train_dataset.flux_frame['CM3up[W/m2]'].to_numpy())
    weighted_mse = WeightedMse(weights, bounds)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # lr_scheduler.step() calls every batch
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                 first_cycle_steps=128 * steps_per_epoch_train,
                                                 cycle_mult=1.5,
                                                 max_lr=1e-4,
                                                 min_lr=5e-7,
                                                 warmup_steps=steps_per_epoch_train if use_warmup else 0,
                                                 gamma=0.8,
                                                 last_epoch=-1)

    train_batch_factory = BatchFactory(train_dataset, cuda_device, preprocess_worker_number=6)
    validation_batch_factory = BatchFactory(val_dataset, cuda_device, preprocess_worker_number=15)
    hard_mining_batch_factory = BatchFactory(hard_mining_dataset, cuda_device, preprocess_worker_number=15)

    best_val_loss = float('Inf')
    best_val_epoch = -1

    try:
        for epoch in range(max_epochs):
            print(f'Epoch {epoch} / {max_epochs}')
            train_loss = train_single_epoch(model,
                                            optimizer,
                                            weighted_mse,
                                            train_batch_factory.cuda_queue,
                                            steps_per_epoch_train,
                                            current_epoch=epoch,
                                            logger=logger,
                                            lr_scheduler=lr_scheduler)
            logger.tb_writer.add_scalar('train_loss_per_epoch', train_loss, epoch)

            val_loss = validate_single_epoch(model,
                                             mse,
                                             validation_batch_factory.cuda_queue,
                                             steps_per_epoch_valid,
                                             current_epoch=epoch,
                                             logger=logger)
            logger.tb_writer.add_scalar('val_loss', val_loss, epoch)
            print(f'Validation loss: {val_loss}')

            if best_val_loss > val_loss:
                # save new model
                torch.save(model, os.path.join(logger.misc_dir, f'model_ep{epoch}.pt'))
                # delete old model
                old_model_path = os.path.join(logger.misc_dir, f'model_ep{best_val_epoch}.pt')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                best_val_loss = val_loss
                best_val_epoch = epoch

            # every single pass of whole dataset, not at epoch == 0
            if epoch % int(len(train_dataset) / train_dataset.batch_size / steps_per_epoch_train + 1) == 0 and epoch:
                calculate_hard_mining_weights(model,
                                              train_dataset,
                                              hard_mining_dataset,
                                              hard_mining_batch_factory.cuda_queue,
                                              logger,
                                              epoch)

    except KeyboardInterrupt:
        pass

    for i in (
            train_batch_factory,
            hard_mining_batch_factory,
            validation_batch_factory,):
        i.stop()

    print('train_model: done')
