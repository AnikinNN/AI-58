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


class CrossEntropyLoss(Metric):
    def __init__(self, label_field: str):
        super().__init__()
        if FluxBatch.is_field_valid(label_field):
            self.label_field = label_field
        else:
            raise ValueError
        self.base_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, model_output: torch.Tensor, batch: FluxBatch):
        return self.base_loss(model_output, batch.true_radiation_class)


class Accuracy(Metric):
    def __init__(self, label_field: str):
        super().__init__()
        if FluxBatch.is_field_valid(label_field):
            self.label_field = label_field
        else:
            raise ValueError

    def __call__(self, model_output: torch.Tensor, batch: FluxBatch):
        _, predicted_labels = torch.max(model_output, dim=1)
        _, true_labels = torch.max(getattr(batch, self.label_field), dim=1)
        labels_comparison = torch.where(predicted_labels == true_labels, 1.0, 0.0)
        return labels_comparison.mean()
