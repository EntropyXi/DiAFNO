# 用途：实现 EDM 扩散训练目标与迭代采样过程。
# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 08/01/2024
# This code is developed with reference to the following GitHub repo:
# [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
# ---------------------------------------------------------------------------------------------

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, reduce

# helpers

# 用途：判断对象是否非 None 的工具函数。
# 参数：输入 val（任意对象）；输出 布尔值。
def exists(val):
    return val is not None

# 用途：val 为 None 时回退到默认值的工具函数。
# 参数：输入 val（可能为 None 的值）、d（默认值或无参工厂）；输出 val 或 d 的结果。
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

# 用途：安全取对数（钳到下限 eps）。
# 参数：输入 t（正张量）、eps（对数下限，默认 1e-20）；输出 log(t)。
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# main class

class ElucidatedDiffusion(nn.Module):
    # 用途：装配 EDM：登记主干与 σ 调度、训练分布及随机采样 churn 超参。
    # 参数：输入 net（去噪主干）、channels（目标通道数）、image_size_h/w/z（空间尺寸）、num_sample_steps（默认采样步数）、sigma_min/sigma_max/sigma_data/rho（σ 调度参数）、P_mean/P_std（训练 σ 对数正态分布）、S_churn/S_tmin/S_tmax/S_noise（随机采样 churn 参数）；输出 无（构造模块状态）。
    def __init__(
            self,
            net,
            *,
            image_size_h,
            image_size_w,
            image_size_z,
            channels=15,
            num_sample_steps=32,
            sigma_min=0.002,
            sigma_max=80,
            sigma_data=1.0,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=0,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
        ):
        super().__init__()

        self.net = net

        self.channels = channels
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
        self.image_size_z = image_size_z

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    # 用途：property：返回主干参数所在设备。
    # 参数：无输入；输出 torch.device。
    def device(self):
        return next(self.net.parameters()).device

    ##### derived preconditioning params - Table 1

    # 用途：EDM 预条件系数 c_skip = σ_data²/(σ²+σ_data²)。
    # 参数：输入 sigma（σ 张量）；输出 同形 c_skip 系数。
    def c_skip(self, sigma):
        return (
            self.sigma_data ** 2
            / (sigma ** 2 + self.sigma_data ** 2)
        )

    # 用途：EDM 预条件系数 c_out = σ·σ_data/√(σ²+σ_data²)。
    # 参数：输入 sigma（σ 张量）；输出 同形 c_out 系数。
    def c_out(self, sigma):
        return (
            sigma
            * self.sigma_data
            * (self.sigma_data ** 2 + sigma ** 2) ** -0.5
        )

    # 用途：EDM 预条件系数 c_in = 1/√(σ²+σ_data²)。
    # 参数：输入 sigma（σ 张量）；输出 同形 c_in 系数。
    def c_in(self, sigma):
        return (
            1
            * (sigma ** 2 + self.sigma_data ** 2) ** -0.5
        )

    # 用途：计算 σ 的时间嵌入输入 c_noise = ln(σ)/4。
    # 参数：输入 sigma（σ 张量）；输出 0.25*log(σ)。
    def c_noise(self, sigma):
        return log(sigma) * 0.25

    ##### preconditioned network output

    # 用途：计算预条件去噪输出 D_θ = c_skip·x + c_out·F_θ(c_in·x, c_noise)。
    # 参数：输入 noised_target（加噪目标 x）、sigma（σ，标量或 [B]）、condition（条件场）；输出 与目标同形的去噪预测。
    def preconditioned_network_forward(
            self,
            noised_target,
            sigma,
            condition,
        ):

        batch = noised_target.shape[0]
        device = noised_target.device

        if isinstance(sigma, float):
            sigma = torch.full(
                (batch,),
                sigma,
                device=device
            )

        padded_sigma = rearrange(
            sigma,
            'b -> b 1 1 1 1'
        )

        net_out = self.net(
            self.c_in(padded_sigma) * noised_target,
            self.c_noise(sigma),
            condition
        )

        out = (
            self.c_skip(padded_sigma) * noised_target
            + self.c_out(padded_sigma) * net_out
        )

        return out

    ##### sampling schedule

    # 用途：生成 ρ=7 的 σ 指数递减采样序列，并在末端补 0。
    # 参数：输入 num_sample_steps（步数，缺省用构造值）；输出 形状 [N+1] 的 σ 序列（首=σ_max，末=0）。
    def sample_schedule(self, num_sample_steps=None):

        num_sample_steps = default(
            num_sample_steps,
            self.num_sample_steps
        )

        if num_sample_steps < 2:
            raise ValueError(
                "num_sample_steps must be at least 2"
            )

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sample_steps,
            device=self.device,
            dtype=torch.float32
        )

        sigmas = (
            self.sigma_max ** inv_rho
            + steps / (N - 1)
            * (
                self.sigma_min ** inv_rho
                - self.sigma_max ** inv_rho
            )
        ) ** self.rho

        sigmas = F.pad(
            sigmas,
            (0, 1),
            value=0.
        )

        return sigmas

    @torch.no_grad()
    # 用途：EDM 随机采样：从 σ_max 高斯噪声出发，逐步 churn 噪声注入 + 二阶 Heun 修正去噪。
    # 参数：输入 condition（条件场 [B,cond_chans,H,W,Z]）、num_sample_steps（步数，覆盖默认）、seed（可复现随机种子）；输出 采样的目标场 [B,channels,H,W,Z]。
    def sample(
            self,
            condition,
            num_sample_steps=None,
            seed=None,
        ):

        if condition.ndim != 5:
            raise ValueError(
                f"condition must have shape [B,C,H,W,Z], "
                f"but got {condition.shape}"
            )

        batch_size = condition.shape[0]

        if condition.shape[2:] != (
                self.image_size_h,
                self.image_size_w,
                self.image_size_z
            ):
            raise ValueError(
                f"condition spatial shape must be "
                f"{(self.image_size_h, self.image_size_w, self.image_size_z)}, "
                f"but got {condition.shape[2:]}"
            )

        num_sample_steps = default(
            num_sample_steps,
            self.num_sample_steps
        )

        shape = (
            batch_size,
            self.channels,
            self.image_size_h,
            self.image_size_w,
            self.image_size_z
        )

        generator = None

        if seed is not None:
            generator = torch.Generator(
                device=self.device
            )
            generator.manual_seed(seed)

        sigmas = self.sample_schedule(
            num_sample_steps
        )

        gammas = torch.where(
            (
                (sigmas >= self.S_tmin)
                & (sigmas <= self.S_tmax)
            ),
            min(
                self.S_churn / num_sample_steps,
                sqrt(2) - 1
            ),
            0.
        )

        sigmas_and_gammas = list(
            zip(
                sigmas[:-1],
                sigmas[1:],
                gammas[:-1]
            )
        )

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(
            shape,
            device=self.device,
            dtype=condition.dtype,
            generator=generator
        )

        for sigma, sigma_next, gamma in tqdm(
                sigmas_and_gammas,
                desc='sampling time step',
                disable=True
            ):

            sigma, sigma_next, gamma = map(
                lambda t: t.item(),
                (sigma, sigma_next, gamma)
            )

            eps = self.S_noise * torch.randn(
                shape,
                device=self.device,
                dtype=condition.dtype,
                generator=generator
            )

            sigma_hat = sigma + gamma * sigma

            images_hat = (
                images
                + sqrt(
                    sigma_hat ** 2
                    - sigma ** 2
                ) * eps
            )

            model_output = (
                self.preconditioned_network_forward(
                    images_hat,
                    sigma_hat,
                    condition
                )
            )

            denoised_over_sigma = (
                images_hat - model_output
            ) / sigma_hat

            images_next = (
                images_hat
                + (
                    sigma_next - sigma_hat
                ) * denoised_over_sigma
            )

            ##### second order correction

            if sigma_next != 0:

                model_output_next = (
                    self.preconditioned_network_forward(
                        images_next,
                        sigma_next,
                        condition
                    )
                )

                denoised_prime_over_sigma = (
                    images_next
                    - model_output_next
                ) / sigma_next

                images_next = (
                    images_hat
                    + 0.5
                    * (
                        sigma_next
                        - sigma_hat
                    )
                    * (
                        denoised_over_sigma
                        + denoised_prime_over_sigma
                    )
                )

            images = images_next

        return images

    ##### training

    # 用途：EDM 损失权重 λ(σ) = (σ²+σ_data²)/(σ·σ_data)²。
    # 参数：输入 sigma（σ 张量）；输出 同形权重。
    def loss_weight(self, sigma):
        return (
            sigma ** 2 + self.sigma_data ** 2
        ) * (
            sigma * self.sigma_data
        ) ** -2

    # 用途：从 LogNormal(P_mean, P_std) 采样训练噪声强度。
    # 参数：输入 batch_size（批大小）；输出 形状 [B] 的 σ。
    def noise_distribution(self, batch_size):
        return (
            self.P_mean
            + self.P_std
            * torch.randn(
                (batch_size,),
                device=self.device
            )
        ).exp()

    # 用途：EDM 训练前向：加噪、预条件去噪、有效像素上的均方误差并乘 λ(σ)。
    # 参数：输入 target（目标场 [B,C,H,W,Z]）、condition（条件场）、target_mask（有效像素 mask，可选，广播到 C）；输出 标量平均损失。
    def forward(
            self,
            target,
            condition,
            target_mask=None,
        ):

        if target.ndim != 5:
            raise ValueError(
                f"target must have shape [B,C,H,W,Z], "
                f"but got {target.shape}"
            )

        if condition.ndim != 5:
            raise ValueError(
                f"condition must have shape [B,C,H,W,Z], "
                f"but got {condition.shape}"
            )

        batch_size, c, h, w, z = target.shape

        if c != self.channels:
            raise ValueError(
                f"expected {self.channels} target channels, "
                f"but got {c}"
            )

        if (h, w, z) != (
                self.image_size_h,
                self.image_size_w,
                self.image_size_z
            ):
            raise ValueError(
                f"target spatial shape must be "
                f"{(self.image_size_h, self.image_size_w, self.image_size_z)}, "
                f"but got {(h, w, z)}"
            )

        if condition.shape[0] != batch_size:
            raise ValueError(
                "target and condition batch sizes do not match"
            )

        if condition.shape[2:] != target.shape[2:]:
            raise ValueError(
                "target and condition spatial shapes do not match"
            )

        sigmas = self.noise_distribution(
            batch_size
        )

        padded_sigmas = rearrange(
            sigmas,
            'b -> b 1 1 1 1'
        )

        noise = torch.randn_like(target)

        noised_target = (
            target
            + padded_sigmas * noise
        )

        denoised = (
            self.preconditioned_network_forward(
                noised_target,
                sigmas,
                condition
            )
        )

        losses = (denoised - target) ** 2

        if target_mask is not None:

            if target_mask.ndim != 5:
                raise ValueError(
                    f"target_mask must have shape [B,C,H,W,Z], "
                    f"but got {target_mask.shape}"
                )

            target_mask = target_mask.to(
                device=target.device,
                dtype=target.dtype
            )

            if (
                    target_mask.shape[1] == 1
                    and target.shape[1] != 1
                ):
                target_mask = target_mask.expand(
                    -1,
                    target.shape[1],
                    -1,
                    -1,
                    -1
                )

            if target_mask.shape != target.shape:
                raise ValueError(
                    f"target_mask shape {target_mask.shape} "
                    f"does not match target shape {target.shape}"
                )

            reduce_dims = tuple(
                range(1, losses.ndim)
            )

            valid_count = target_mask.sum(
                dim=reduce_dims
            ).clamp_min(1.0)

            losses = (
                losses * target_mask
            ).sum(
                dim=reduce_dims
            ) / valid_count

        else:

            losses = reduce(
                losses,
                'b ... -> b',
                'mean'
            )

        losses = (
            losses
            * self.loss_weight(sigmas)
        )

        return losses.mean()
