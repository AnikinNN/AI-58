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
        self.images = []
        self.elevations = []
        self.fluxes = []
        self.hard_mining_weights = []
        self.train_frame_indexes = []

        self.state = FluxBatchState.CPU_APPENDING

    def __len__(self):
        return len(self.train_frame_indexes)

    def append(self, image: torch.Tensor, elevation: float,
               flux: float, hard_mining_weight: float, train_frame_index: int):
        if self.state is not FluxBatchState.CPU_APPENDING:
            raise ValueError(f'You can append to batch only on FluxBatchState.CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.images.append(image)
        self.elevations.append(elevation)
        self.fluxes.append(flux)
        self.hard_mining_weights.append(hard_mining_weight)
        self.train_frame_indexes.append(train_frame_index)

    def to_tensor(self):
        if self.state is not FluxBatchState.CPU_APPENDING:
            raise ValueError(f'You can convert to tensor only from FluxBatchState.CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.fluxes = torch.reshape(torch.tensor(self.fluxes), (-1, 1))
        self.images = torch.stack(self.images)
        # may be useful for moving augmentation to GPU
        # self.images = torch.stack(tuple((i / 255.).transpose(2, 0, 1) for i in self.images))
        self.elevations = torch.reshape(torch.tensor(self.elevations), (-1, 1))
        self.hard_mining_weights = np.array(self.hard_mining_weights)
        self.state = FluxBatchState.CPU_STORING

    def to_cuda(self, cuda_device, to_variable: bool):
        if self.state is not FluxBatchState.CPU_STORING:
            raise ValueError(f'You can load to cuda device only on FluxBatchState.CPU_STORING state. '
                             f'But there was an attempt on {self.state} state')

        self.fluxes = self.fluxes.float()
        self.images = self.images.float()
        self.elevations = self.elevations.float()

        if to_variable:
            self.fluxes = Variable(self.fluxes)
            self.images = Variable(self.images)
            self.elevations = Variable(self.elevations)

        self.fluxes = self.fluxes.to(cuda_device)
        self.images = self.images.to(cuda_device)
        self.elevations = self.elevations.to(cuda_device)
        self.state = FluxBatchState.CUDA_STORING
