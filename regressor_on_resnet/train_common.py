import warnings
from typing import Union
import numpy as np
import os
import torch
from tqdm import tqdm
from queue import Queue

from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.sgdr_restarts_warmup import CosineAnnealingWarmupRestarts
from regressor_on_resnet.batch_factory import BatchFactory
from regressor_on_resnet.normalizer import Normalizer
from regressor_on_resnet.resnet_regressor import ResnetRegressor


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
    loss_history = []
    pbar = tqdm(total=per_step_epoch, ncols=100)
    pbar.set_description(desc='train')

    warning_elapsed = False

    for batch_idx in range(per_step_epoch):
        batch = cuda_batches_queue.get(block=True)

        if batch_idx == 0 and current_epoch == 0:
            logger.store_batch_as_image('train_batch', batch.images,
                                        global_step=current_epoch,
                                        inv_normalizer=Normalizer.inv_normalizer)

        optimizer.zero_grad()
        data_out = model(batch.images, batch.elevations)
        loss = loss_function(data_out, batch.fluxes)
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})

        logger.tb_writer.add_scalar('train_loss_per_step', loss_history[-1], current_epoch * per_step_epoch + batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step()
        else:
            if not warning_elapsed:
                warnings.warn('lr_scheduler is None')
                warning_elapsed = True

    pbar.close()

    loss_mean = np.mean(loss_history)
    logger.tb_writer.add_scalar('train_loss_per_epoch', loss_mean, current_epoch)

    return loss_mean


def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module,
                          quality_function: torch.nn.Module,
                          cuda_batches_queue: Queue,
                          per_step_epoch: int,
                          current_epoch: int,
                          logger: Logger,
                          ):
    model.eval()
    loss_values = []
    qualities = []

    pbar = tqdm(total=per_step_epoch, ncols=100)
    pbar.set_description(desc='validation')
    for batch_idx in range(per_step_epoch):
        batch = cuda_batches_queue.get(block=True)

        with torch.no_grad():
            data_out = model(batch.images, batch.elevations)

        if batch_idx == 0 and current_epoch == 0:
            logger.store_batch_as_image('val_batch', batch.images,
                                        global_step=current_epoch,
                                        inv_normalizer=Normalizer.inv_normalizer)

        loss = loss_function(data_out, batch.fluxes)
        loss_values.append(loss.item())

        quality = quality_function(data_out, batch.fluxes)
        qualities.append(quality.item())

        pbar.update()
        pbar.set_postfix({'loss': loss_values[-1], 'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    return np.mean(loss_values), np.mean(qualities)


def train_model(model: ResnetRegressor,
                loss,
                train_batch_factory: BatchFactory,
                validation_batch_factory: BatchFactory,
                logger: Logger,
                max_epochs: int,
                steps_per_epoch_train: int,
                steps_per_epoch_valid: int,
                train_convolutional_since_epoch: int,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: Union[None, CosineAnnealingWarmupRestarts]):
    best_val_loss = float('Inf')
    best_val_epoch = -1

    try:
        for epoch in range(max_epochs):
            print(f'Epoch {epoch} / {max_epochs}')

            # enable training of convolutional part at desired epoch
            if train_convolutional_since_epoch is not None and train_convolutional_since_epoch == epoch:
                print('train convolutional on')
                model.set_train_convolutional_part(True)

            train_loss = train_single_epoch(model,
                                            optimizer,
                                            loss,
                                            train_batch_factory.cuda_queue,
                                            steps_per_epoch_train,
                                            current_epoch=epoch,
                                            logger=logger,
                                            lr_scheduler=lr_scheduler)

            val_weighted_mse, val_mse = validate_single_epoch(model,
                                                              loss,
                                                              loss,
                                                              validation_batch_factory.cuda_queue,
                                                              steps_per_epoch_valid,
                                                              current_epoch=epoch,
                                                              logger=logger)
            logger.tb_writer.add_scalar('val_mse', val_mse, epoch)
            logger.tb_writer.add_scalar('val_weighted_mse', val_weighted_mse, epoch)
            print(f'Validation loss: {val_mse}')

            if best_val_loss > val_mse:
                # save new model
                torch.save(model, os.path.join(logger.misc_dir, f'model_ep{epoch}.pt'))
                # delete old model
                old_model_path = os.path.join(logger.misc_dir, f'model_ep{best_val_epoch}.pt')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                # update best values
                best_val_loss = val_mse
                best_val_epoch = epoch

    except KeyboardInterrupt:
        pass

    finally:
        for i in (
                train_batch_factory,
                validation_batch_factory,):
            i.stop()

    print('train_model: done')
    return best_val_loss
