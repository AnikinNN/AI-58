import torch

from regressor_on_resnet.batch import FluxBatch
from regressor_on_resnet.resnet_regressor import ResnetRegressor, ResnetClassifier


def test_resnet_regressor():
    model = ResnetRegressor((1, 2, 3))

    assert model.tail.__len__() == 4 * 2

    batch = FluxBatch()
    batch.images = torch.rand(32, 3, 512, 512)
    batch.elevations = torch.rand(32, 1)

    model_output = model(batch)
    assert model_output.shape == (32, 1)


def test_resnet_classifier():
    model = ResnetClassifier((1, 2, 3), 10)

    assert model.tail.__len__() == 4 * 2 - 1

    batch = FluxBatch()
    batch.images = torch.rand(32, 3, 512, 512)
    batch.elevations = torch.rand(32, 1)

    model_output = model(batch)
    assert model_output.shape == (32, 10)
    assert model_output.min() < 0
