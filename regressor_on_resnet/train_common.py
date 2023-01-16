from typing import Union
import numpy as np
import os
import torch
from tqdm import tqdm

from regressor_on_resnet.metrics import Metric
from regressor_on_resnet.nn_logging import Logger, BestModelSaver
from regressor_on_resnet.sgdr_restarts_warmup import CosineAnnealingWarmupRestarts
from regressor_on_resnet.batch_factory import BatchFactory
from regressor_on_resnet.normalizer import Normalizer
from regressor_on_resnet.resnet_regressor import ResnetRegressor


class Trainer:
    def __init__(self,
                 model: ResnetRegressor,
                 loss: Metric,
                 validation_metrics: list[Metric],
                 train_batch_factory: BatchFactory,
                 validation_batch_factory: BatchFactory,
                 logger: Logger,
                 max_epochs: int,
                 steps_per_epoch_train: int,
                 steps_per_epoch_valid: int,
                 train_convolutional_since_epoch: int,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: Union[None, CosineAnnealingWarmupRestarts],
                 ):
        self.model = model
        self.loss = loss
        self.validation_metrics = validation_metrics
        self.train_batch_factory = train_batch_factory
        self.validation_batch_factory = validation_batch_factory
        self.logger = logger
        self.max_epochs = max_epochs
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_valid = steps_per_epoch_valid
        self.train_convolutional_since_epoch = train_convolutional_since_epoch
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_model(self):
        try:
            for epoch in range(self.max_epochs):
                print(f'Epoch {epoch} / {self.max_epochs}')

                # enable training of convolutional part at desired epoch
                if self.train_convolutional_since_epoch is not None and self.train_convolutional_since_epoch == epoch:
                    print('train convolutional on')
                    self.model.set_train_convolutional_part(True)

                train_loss = self.train_single_epoch(current_epoch=epoch)
                print(f'Train loss: {train_loss}')

                validation_result = self.validate_single_epoch(current_epoch=epoch)
                print(f'Validation result: {validation_result}')

                val_mse = validation_result[0]
                self.logger.save_model(self.model, val_mse, epoch)

        except KeyboardInterrupt:
            pass

        finally:
            for i in (
                    self.train_batch_factory,
                    self.validation_batch_factory,):
                i.stop()

        print('train_model: done')
        return self.logger.best_model_saver.best_val_metric

    def train_single_epoch(self, current_epoch: int):
        self.model.train()
        loss_history = []
        pbar = tqdm(total=self.steps_per_epoch_train, ncols=100)
        pbar.set_description(desc='train')

        for batch_idx in range(self.steps_per_epoch_train):
            batch = self.train_batch_factory.cuda_queue.get(block=True)

            if batch_idx == 0 and current_epoch == 0:
                self.logger.store_batch_as_image('train_batch', batch.images,
                                                 global_step=current_epoch,
                                                 inv_normalizer=Normalizer.inv_normalizer)

            self.optimizer.zero_grad()
            model_output = self.model(batch)
            loss = self.loss(model_output, batch)
            loss_history.append(loss.item())

            loss.backward()
            self.optimizer.step()

            pbar.update()
            pbar.set_postfix({
                self.loss.__class__.__name__: loss.item(),
                'cuda_queue_len': self.train_batch_factory.cuda_queue.qsize()
            })

            self.logger.tb_writer.add_scalar('train_loss_per_step', loss_history[-1],
                                             current_epoch * self.steps_per_epoch_train + batch_idx)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        pbar.close()

        loss_mean = np.mean(loss_history)
        self.logger.tb_writer.add_scalar('train_loss_per_epoch', loss_mean, current_epoch)

        return loss_mean

    def validate_single_epoch(self, current_epoch: int):
        self.model.eval()
        metrics_values = [[] for _ in self.validation_metrics]

        pbar = tqdm(total=self.steps_per_epoch_valid, ncols=100)
        pbar.set_description(desc='validation')

        for batch_idx in range(self.steps_per_epoch_valid):
            batch = self.validation_batch_factory.cuda_queue.get(block=True)

            with torch.no_grad():
                model_output = self.model(batch)

            if batch_idx == 0 and current_epoch == 0:
                self.logger.store_batch_as_image('val_batch', batch.images,
                                                 global_step=current_epoch,
                                                 inv_normalizer=Normalizer.inv_normalizer)

            for i, metric in enumerate(self.validation_metrics):
                metrics_values[i].append(metric(model_output, batch).item())

            pbar.update()
            metrics_pbar = {f'validation {self.validation_metrics[i].__class__.__name__}': metrics_values[i][-1]
                            for i in range(len(self.validation_metrics))}
            pbar.set_postfix({**metrics_pbar, 'cuda_queue_len': self.validation_batch_factory.cuda_queue.qsize()})
        pbar.close()

        metrics_values = [np.mean(i) for i in metrics_values]

        for i in range(len(self.validation_metrics)):
            self.logger.tb_writer.add_scalar(
                self.validation_metrics[i].__class__.__name__, metrics_values[i], current_epoch)

        return metrics_values
