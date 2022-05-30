import warnings

import numpy as np
import torch


def get_weights_and_bounds(target: np.ndarray):
    bounds = np.percentile(target, np.arange(0, 101))
    weights = bounds[1:] - bounds[:-1]
    # weights = np.histogram(target, bins=bounds, density=True)
    return weights / weights.mean(), bounds[1:-1]


class WeightedMse:
    def __init__(self, weights: np.ndarray, bounds: np.ndarray):
        for key, val in {'weights': weights, 'bounds': bounds}.items():
            if len(val.shape) > 1:
                warnings.warn(f'{key} has shape of {val.shape}. It will be flattened')

        weights = weights.flatten()
        bounds = bounds.flatten()

        assert weights.shape[0] - bounds.shape[0] == 1, 'weights must be 1 more than bounds'

        self.weights = weights
        self.bounds = bounds

    def __call__(self, predicted, target, hard_mining_weights):
        return self.weighted_mse_loss(predicted, target, hard_mining_weights)

    def weighted_mse_loss(self, predicted: torch.Tensor, target: torch.Tensor, hard_mining_weights: np.ndarray):
        target_ndarray = target.cpu().detach().numpy().flatten()
        weights_on_input = self.get_weights_on_input(target_ndarray)
        weights_on_input = torch.tensor(weights_on_input.reshape((-1, 1))).to(predicted.get_device())
        hard_mining_weights = torch.tensor(hard_mining_weights.reshape((-1, 1))).to(predicted.get_device())
        return (torch.square(predicted - target) * weights_on_input * hard_mining_weights).mean()

    def get_weights_on_input(self, target):
        weight_index = np.searchsorted(self.bounds, target, side='right')

        return self.weights[weight_index]


if __name__ == '__main__':
    # weights_ = np.arange(20, 25, 1)
    # bounds_ = np.arange(10, 14, 1)
    # target_ = np.arange(9, 15, 0.5)
    # loss_func = WeightedMse(weights_, bounds_)
    # print(target_, '\n', weights_, '\n', bounds_)
    # print(loss_func.get_weights_on_input(target_))

    target_ = np.random.rand(1000000) * 1200
    weights_, bounds_ = get_weights_and_bounds(target_)
    loss_func = WeightedMse(weights_, bounds_)

