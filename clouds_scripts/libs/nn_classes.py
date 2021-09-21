import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import collections
import threading
from sklearn.utils import shuffle


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_objects_i(objects_count):
    """Cyclic generator of paths indices
    """
    current_objects_id = 0
    while True:
        yield current_objects_id
        current_objects_id = (current_objects_id + 1) % objects_count


class DS(Dataset):
    def __init__(self, X, y, augment = True, shuffle = True):
        super(DS, self).__init__()
        
        assert X.shape[0] == y.shape[0], 'X and y should be of the same length.'
        self.X = np.concatenate([X, y.reshape((-1,1))], axis=1)
        self.augment = augment
        self.shuffle = shuffle

        self.objects_id_generator = threadsafe_iter(get_objects_i(self.X.shape[0]))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0

    def __len__(self):
        return self.X.shape[0]

    def shuffle_data(self):
        self.X = shuffle(self.X)

    def __getitem__(self, obj_id):
        obj_features = self.X[obj_id,:].reshape((1,-1))
        obj_X = obj_features[:,:-1].reshape((-1,))
        obj_y = obj_features[:,-1].reshape((-1,))
        return obj_X, obj_y


class MSE(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(MSE, self).__init__()
        self.reduction = reduction
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        loss = torch.mean(torch.square(output - target))
        
        return loss


class MAPE(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(MAPE, self).__init__()
        self.reduction = reduction
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        loss = torch.mean(torch.abs((output - target)/(torch.abs(target)+1e-8)))
        return loss


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishAutoFn.apply(x)

class MishAutoFn(torch.autograd.Function):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.mul(torch.tanh(nn.Softplus()(x)))  # x * tanh(ln(1 + exp(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        x_tanh_sp = nn.Softplus()(x).tanh()
        return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


