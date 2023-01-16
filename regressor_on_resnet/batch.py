from enum import Enum

import numpy as np
import torch
from torch.autograd import Variable


class FluxBatchState(Enum):
    CPU_APPENDING = 1
    CPU_STORING = 2
    CUDA_STORING = 3


class FluxBatch:
    def __init__(self):
        self.tensor_attributes = [
            'images',
            'masks',
            'elevations',
            'fluxes'
        ]

        self.images = []
        self.masks = []
        self.elevations = []
        self.fluxes = []

        self.hard_mining_weights = []
        self.train_frame_indexes = []

        self.true_cloud_class = []
        self.true_radiation_class = []
        self.predicted_cloud_class = []
        self.predicted_radiation_class = []

        self.state = FluxBatchState.CPU_APPENDING
        self.len = 0

    def __len__(self):
        return self.len

    def append(self, **kwargs) -> None:
        """
        Appends every item in kwargs to a list by keyword.
        Keyword must be a Batch object's attribute which has an append() method
        """
        if self.state is not FluxBatchState.CPU_APPENDING:
            raise ValueError(f'You can append to batch only on FluxBatchState.CPU_APPENDING state. '
                             f'But there was an appending attempt on {self.state} state')

        self.len += 1

        for key, value in kwargs.items():
            attribute = self.__getattribute__(key)
            attribute.append(value)

            # ensure that every append() call same lists change
            assert len(attribute) == self.len, \
                f'There was an attempt to append to self.{key} but it\'s len not equal to {self.len}.\n' \
                f'Probably different attributes were appended now and before'

    def to_tensor(self) -> None:
        """
        Converts list attributes to tensors by stacking them along new axis
        """
        if self.state is not FluxBatchState.CPU_APPENDING:
            raise ValueError(f'You can convert to tensor only from FluxBatchState.CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        for i in self.tensor_attributes:
            attribute = self.__getattribute__(i)
            if len(attribute):
                tensor = torch.tensor(np.stack(attribute, axis=0))

                # in case of single number value
                if tensor.shape == (self.__len__(),):
                    tensor = tensor.unsqueeze(dim=1)

                self.__setattr__(i, tensor)
            else:
                self.__setattr__(i, None)

        self.state = FluxBatchState.CPU_STORING

    def _to_tensor(self):
        if self.state is not FluxBatchState.CPU_APPENDING:
            raise ValueError(f'You can convert to tensor only from FluxBatchState.CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.images = torch.stack(tuple(torch.tensor((i / 255.).transpose(2, 0, 1)) for i in self.images))
        self.masks = torch.stack(tuple(torch.tensor(i.transpose(2, 0, 1)) for i in self.masks))
        self.elevations = torch.reshape(torch.tensor(self.elevations), (-1, 1))
        self.fluxes = torch.reshape(torch.tensor(self.fluxes), (-1, 1))
        self.hard_mining_weights = np.array(self.hard_mining_weights)

        self.state = FluxBatchState.CPU_STORING

    def to_cuda(self, to_variable: bool, cuda_device: torch.device) -> None:
        if self.state is not FluxBatchState.CPU_STORING:
            raise ValueError(f'You can load to cuda device only on FluxBatchState.CPU_STORING state. '
                             f'But there was an attempt on {self.state} state')

        for i in self.tensor_attributes:
            attribute = self.__getattribute__(i)
            if attribute is not None:
                attribute = attribute.float()
                if to_variable:
                    attribute = Variable(attribute)
                attribute = attribute.to(cuda_device)
                self.__setattr__(i, attribute)

        self.state = FluxBatchState.CUDA_STORING

    def _to_cuda(self, cuda_device, to_variable: bool):
        if self.state is not FluxBatchState.CPU_STORING:
            raise ValueError(f'You can load to cuda device only on FluxBatchState.CPU_STORING state. '
                             f'But there was an attempt on {self.state} state')

        self.fluxes = self.fluxes.float()
        self.images = self.images.float()
        self.masks = self.masks.float()
        self.elevations = self.elevations.float()

        if to_variable:
            self.fluxes = Variable(self.fluxes)
            self.images = Variable(self.images)
            self.masks = Variable(self.masks)
            self.elevations = Variable(self.elevations)

        self.fluxes = self.fluxes.to(cuda_device)
        self.images = self.images.to(cuda_device)
        self.masks = self.masks.to(cuda_device)
        self.elevations = self.elevations.to(cuda_device)

        self.state = FluxBatchState.CUDA_STORING
