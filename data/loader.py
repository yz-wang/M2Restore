import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import copy
import random

from utils import mprint

def create_loader(dataset: Dataset, ds_name: str, task_name: str, global_seed: int,
                  batch_size: int, shuffle: bool = False, num_workers: int = 0, ):
    
    mprint('[{}] dataset_length: [{}]'.format(ds_name, len(dataset)))
    loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True)

    return loader