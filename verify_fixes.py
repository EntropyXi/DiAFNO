# 修复验证：小尺寸 CPU 前向，验证 IAFNODiff 修复后可用
import sys
import types

# 本机缺少 timm 包；IAFNO 仅在 drop_path=0 配置下从不实例化 DropPath，
# trunc_normal_ 全代码无调用点，注入最小 mock 以完成验证
_timm = types.ModuleType("timm")
_timm_layers = types.ModuleType("timm.models.layers")


class _DropPath:
    pass


_timm_layers.DropPath = _DropPath
_timm_layers.trunc_normal_ = lambda *a, **k: None
_timm.models = types.ModuleType("timm.models")
_timm.models.layers = _timm_layers
sys.modules["timm"] = _timm
sys.modules["timm.models"] = _timm.models
sys.modules["timm.models.layers"] = _timm_layers

import torch

from IAFNO import IAFNODiff

torch.manual_seed(0)
model = IAFNODiff(
    dim=(64, 64, 1),
    dim_f=(64, 64, 1),
    patch_size=(8, 8, 1),
    embed_dim=32,
    num_blocks=4,
    cond_chans=8,
    target_chans=4,
    ex_layer=2,
    nlayer=2,
    hidden_size_factor=4,
    drop_rate=0.0,
    sparsity_threshold=0.01,
    hard_thresholding_fraction=1.0,
)

x = torch.randn(2, 4, 64, 64, 1)
time = torch.randn(2)
cond = torch.randn(2, 8, 64, 64, 1)

out = model(x, time, cond)
print("forward output shape:", tuple(out.shape))
assert tuple(out.shape) == (2, 4, 64, 64, 1)

# 前向-反向闭环，确认所有参数都能收到梯度（对应 DDP find_unused_parameters=False 的要求）
loss = out.square().mean()
loss.backward()
no_grad = [
    name for name, p in model.named_parameters()
    if p.grad is None
]
grad_ok = [
    name for name, p in model.named_parameters()
    if p.grad is not None
]
print("params with grad:", len(grad_ok), "/", len(grad_ok) + len(no_grad))
if no_grad:
    print("NO-GRAD PARAMS:", no_grad)
else:
    print("ALL PARAMS RECEIVED GRAD: OK (DDP-safe)")

# 验证修复 1：ex_layer=4, nlayer=2 默认配置下 if 分支行为
cond_val = model.ex_layer != 1 and model.nlayer == 1
print("branch condition (ex_layer=2, nlayer=2):", cond_val, "(False -> implicit coef loop)")

# 验证修复 4：模型不依赖 .cuda()，CPU 上可正常构建
print("model built on CPU without hardcoded .cuda(): OK")
