# 用途：初始化设备、随机种子与分布式进程环境。
import os
import random

import numpy as np
import torch
import torch.distributed as dist


class DistributedRuntime:
    def __init__(self):
        self.distributed = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")

    @property
    def is_main_process(self):
        return self.rank == 0

    def setup(self):
        self.world_size = int(
            os.environ.get("WORLD_SIZE", "1")
        )
        self.distributed = self.world_size > 1
        if self.distributed:
            self.local_rank = int(
                os.environ["LOCAL_RANK"]
            )
            self.rank = int(os.environ["RANK"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://"
            )
            self.device = torch.device(
                "cuda",
                self.local_rank
            )
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        return self

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def cleanup(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def set_random_seed(seed, rank):
    current_seed = seed + rank
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
