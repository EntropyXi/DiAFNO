import torch
import torch.distributed as dist


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
