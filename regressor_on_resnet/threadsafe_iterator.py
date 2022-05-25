import threading

import torch
from torch.autograd import Variable


class ThreadsafeIterator:
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


class ThreadKiller:
    """Boolean object for signaling a worker thread to terminate"""
    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_to_kill(self, to_kill):
        self.to_kill = to_kill


def threaded_batches_feeder(to_kill, batches_queue, dataset_generator):
    for img, target in dataset_generator:
        batches_queue.put((img, target), block=True)
        if to_kill():
            print('cpu_feeder_killed')
            return


def threaded_cuda_feeder(to_kill, cuda_batches_queue, batches_queue, cuda_device):
    while not to_kill():
        cuda_device = torch.device(cuda_device)
        (x, flux) = batches_queue.get(block=True)

        flux = torch.tensor(tuple([i] for i in flux))
        img = torch.stack(tuple(i[0] for i in x))
        elevation = torch.tensor(tuple([i[1]] for i in x))

        flux = Variable(flux.float()).to(cuda_device)
        img = Variable(img.float()).to(cuda_device)
        elevation = Variable(elevation.float()).to(cuda_device)
        cuda_batches_queue.put((img, flux, elevation), block=True)
    print('cuda_feeder_killed')
    return

# class BathFactory:
#     def __init__(self, cpu_queue_length, cuda_queue_length):
