import torch
from torch import distributed
from torch.utils.data import Sampler
import random


class DistributedSampler(Sampler):
    """This sampler class is used for distributed computing. Data set is shuffled for each processes."""

    def __init__(self, data_source, group=None) -> None:
        self.data_source = data_source
        self.group = group
        self.rand = distributed.get_global_rank(group)
        self.global_size = distributed.get_world_size(group)

        if self.rank == 0:
            # process 0 shuffle a list, other process get the list
            self.idxs = [i for i in range(len(self.data_source))]
            random.shuffle(self.idxs)
            self.idxs = torch.tensor(self.idxs, dtype=torch.int)
        else:
            self.idxs = torch.zeros(len(self.data_source), dtype=torch.int)

        distributed.barrier(group)
        distributed.all_reduce(self.idxs)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        for i in range(self.rank, len(self.data_source), self.global_size):
            yield self.data_source[i]
