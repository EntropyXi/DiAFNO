# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 08/01/2024
# This code is developed with reference to the following GitHub repo:
# [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
# ---------------------------------------------------------------------------------------------

from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# main class

class ElucidatedDiffusion(nn.Module):
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
            S_churn=80,
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
    def device(self):
        return next(self.net.parameters()).device

    ##### derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (
            self.sigma_data ** 2
            / (sigma ** 2 + self.sigma_data ** 2)
        )

    def c_out(self, sigma):
        return (
            sigma
            * self.sigma_data
            * (self.sigma_data ** 2 + sigma ** 2) ** -0.5
        )

    def c_in(self, sigma):
        return (
            1
            * (sigma ** 2 + self.sigma_data ** 2) ** -0.5
        )

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    ##### preconditioned network output

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

    def loss_weight(self, sigma):
        return (
            sigma ** 2 + self.sigma_data ** 2
        ) * (
            sigma * self.sigma_data
        ) ** -2

    def noise_distribution(self, batch_size):
        return (
            self.P_mean
            + self.P_std
            * torch.randn(
                (batch_size,),
                device=self.device
            )
        ).exp()

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