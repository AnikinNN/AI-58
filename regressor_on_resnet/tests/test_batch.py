import torch
import pytest

from ..batch import FluxBatch


def test_batch_init():
    assert len(FluxBatch()) == 0


def test_batch_lifetime():
    """positive test of batch lifetime"""
    batch = FluxBatch()
    assert len(batch) == 0

    batch_len = 10
    for i in range(batch_len):
        batch.append(images=[69 + i, 420], train_frame_indexes=i)

    assert len(batch) == batch_len
    assert len(batch.images) == batch_len
    assert len(batch.masks) == 0

    batch.to_tensor()

    assert isinstance(batch.images, torch.Tensor)
    assert batch.masks is None
    assert isinstance(batch.train_frame_indexes, list)

    assert batch.images.shape == (batch_len, 2)

    cuda_device = torch.device(0)
    batch.to_cuda(True, cuda_device)

    assert isinstance(batch.images, torch.Tensor)
    assert isinstance(batch.train_frame_indexes, list)

    assert batch.images.get_device() == cuda_device.index


def test_invalid_append():
    """
    negative test of invalid use of append method
    """
    batch = FluxBatch()

    batch.append(masks=[69, 420], fluxes=1)

    with pytest.raises(Exception) as exc_info:
        batch.append(hard_mining_weights=[42], predicted_radiation_class=101)
    assert exc_info.type is AssertionError
