# 用途：初始化设备、随机种子与分布式进程环境。
import os
import random

import numpy as np
import torch
import torch.distributed as dist


class DistributedRuntime:
    # 用途：初始化分布式运行时占位（默认单进程语义）。
    # 参数：无输入；输出 无。
    def __init__(self):
        self.distributed = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")

    @property
    # 用途：判断当前进程是否为主进程（rank 0 或单进程）。
    # 参数：无输入；输出 布尔值。
    def is_main_process(self):
        return self.rank == 0

    # 用途：初始化 DDP 进程组（env://，单卡时退化为无分布式）。
    # 参数：无输入；输出 无。
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

    # 用途：同步屏障（单进程时为空操作）。
    # 参数：无输入；输出 无。
    def barrier(self):
        if self.distributed:
            dist.barrier()

    # 用途：销毁 DDP 进程组。
    # 参数：无输入；输出 无。
    def cleanup(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


# 用途：统一设置 python/numpy/torch(+CUDA) 全局随机种子（按 rank 偏移）。
# 参数：输入 seed（基础种子）、rank（进程序号）；输出 无。
def set_random_seed(seed, rank):
    current_seed = seed + rank
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
