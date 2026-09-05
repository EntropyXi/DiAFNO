# 用途：计算有效像素掩膜下、适配 DDP 全局归一化的 MSE。
import torch
import torch.distributed as dist


# 用途：DDP 正确的全局 mask 均值：本地分子乘 world_size/global_count，使梯度 allreduce 后等价于全局掩膜均值。
# 参数：输入 losses（逐元素平方误差张量）、mask（有效像素 mask）；输出 标量损失（其梯度即全局均值梯度）。
def globally_normalized_masked_mse(losses, mask):
    """Return a DDP-correct masked mean without reducing gradients.

    DDP averages parameter gradients across ranks.  Scaling each rank's
    differentiable numerator by world_size/global_valid_count makes that
    averaged gradient equal to the gradient of the global masked mean.
    """
    mask = mask.to(dtype=losses.dtype)
    local_sum = (losses * mask).sum()
    local_count = mask.sum().detach()
    if dist.is_available() and dist.is_initialized():
        global_count = local_count.clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        return (
            local_sum * world_size
            / global_count.clamp_min(1.0)
        )
    return local_sum / local_count.clamp_min(1.0)
