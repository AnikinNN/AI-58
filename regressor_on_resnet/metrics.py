import torch
from torch.nn import MSELoss

from regressor_on_resnet.batch import FluxBatch


class Metric:
    def __init__(self):
        pass

    def __call__(self, model_output: torch.Tensor, batch: FluxBatch):
        raise NotImplementedError


class MseLoss(Metric):
    def __init__(self):
        super().__init__()
        self.base_loss = MSELoss()

    def __call__(self, model_output: torch.Tensor, batch: FluxBatch):
        return self.base_loss(model_output, batch.fluxes)
