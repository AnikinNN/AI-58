import threading
from queue import Queue

import torch
from torch.autograd import Variable

from regressor_on_resnet.flux_dataset import FluxDataset


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
        row_ids = tuple(i[2] for i in x)

        flux = Variable(flux.float()).to(cuda_device)
        img = Variable(img.float()).to(cuda_device)
        elevation = Variable(elevation.float()).to(cuda_device)
        cuda_batches_queue.put((img, flux, elevation, row_ids), block=True)
    print('cuda_feeder_killed')
    return


class BatchFactory:
    def __init__(self,
                 dataset: FluxDataset,
                 cuda_device,
                 cpu_queue_length=4,
                 cuda_queue_length=4,
                 preprocess_worker_number=4,
                 cuda_feeder_number=1):
        self.cpu_queue = Queue(maxsize=cpu_queue_length)
        self.cuda_queue = Queue(maxsize=cuda_queue_length)

        # one killer for all threads
        self.threads_killer = ThreadKiller()
        self.threads_killer.set_to_kill(False)

        # thread storage to watch after their closing
        self.cuda_feeders = []
        self.preprocess_workers = []

        for _ in range(cuda_feeder_number):
            thr = threading.Thread(target=threaded_cuda_feeder,
                                   args=(self.threads_killer,
                                         self.cuda_queue,
                                         self.cpu_queue,
                                         cuda_device))
            thr.start()
            self.cuda_feeders.append(thr)

        for _ in range(preprocess_worker_number):
            thr = threading.Thread(target=threaded_batches_feeder,
                                   args=(self.threads_killer,
                                         self.cpu_queue,
                                         dataset))
            thr.start()
            self.preprocess_workers.append(thr)

    def stop(self):
        self.threads_killer.set_to_kill(True)

        # clean cuda_queues to stop cuda_feeder
        while sum(map(lambda x: int(x.is_alive()), self.cuda_feeders)):
            while not self.cuda_queue.empty():
                self.cuda_queue.get()

        # clean cpu_queues to stop preprocess_workers
        while sum(map(lambda x: int(x.is_alive()), self.preprocess_workers)):
            while not self.cpu_queue.empty():
                self.cpu_queue.get()
