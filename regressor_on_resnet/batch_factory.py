import threading
from queue import Queue
import torch

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.gpu_augmenter import Augmenter
from regressor_on_resnet.normalizer import Normalizer


class ThreadKiller:
    """Boolean object for signaling a worker thread to terminate"""
    # todo delete class, replace usages with threading.Event

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_to_kill(self, to_kill):
        self.to_kill = to_kill


def threaded_batches_feeder(to_kill: ThreadKiller, target_queue: Queue, dataset_generator: FluxDataset):
    """
    takes batch from dataset_generator and put to target_queue until to_kill
    """
    for batch in dataset_generator:
        target_queue.put(batch, block=True)
        if to_kill():
            print('cpu_feeder_killed')
            return


def threaded_cuda_feeder(to_kill: ThreadKiller,
                         target_queue: Queue,
                         source_queue: Queue,
                         cuda_device: torch.device,
                         to_variable: bool,
                         do_augment: bool):
    """
    takes batch from source_queue, transforms data to tensors, puts to target_queue until to_kill
    """
    while not to_kill():
        batch = source_queue.get(block=True)
        batch.to_tensor()
        batch.to_cuda(to_variable, cuda_device)
        if do_augment:
            batch.images, batch.masks, batch.elevations = Augmenter.call(batch)
        batch.elevations = torch.deg2rad(batch.elevations)
        batch.images = Normalizer.call(batch.images)
        batch.images = batch.images * batch.masks
        target_queue.put(batch, block=True)
    print('cuda_feeder_killed')
    return


class BatchFactory:
    def __init__(self,
                 dataset: FluxDataset,
                 cuda_device: torch.device,
                 do_augment: bool,
                 cpu_queue_length: int,
                 cuda_queue_length: int,
                 preprocess_worker_number: int,
                 cuda_feeder_number: int,
                 to_variable: bool,
                 ):
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
                                         cuda_device,
                                         to_variable,
                                         do_augment)
                                   )
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
