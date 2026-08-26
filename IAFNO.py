# ---------------------------------------------------------------------------------------------
# Author: Yuchi Jiang
# LatestVersionDate: 07/27/2026 (specifically designed for diffusion)
# ---------------------------------------------------------------------------------------------

# Many thanks to all the authors of:
# Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B.: Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. arXiv preprint arXiv:2111.13587 (2021)

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")  # 必须在 pyplot 导入前设置，避免服务器 DISPLAY 环境下卡住
import matplotlib.pyplot as plt
from utilities3 import *


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

# 全局 RNG 种子由训练器 set_random_seed(seed + rank) 统一管理，
# 不在此处硬编码，避免 import 时污染调用方的种子体系

################################################################################################################################

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print(emb.shape)
        return emb

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale # normalize 是 L2 norm 需要补一个*sqrt(dim)转为标准RMSNorm

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 把 3D 空间场切成 3D patch，并把每个 patch 映射成一个 embedding vector
class PatchEmbed(nn.Module):
    def __init__(self, length, patch_size, embed_dim, in_chans):              #####   Length & Patch_size must be 3 dims   #####
        super().__init__()
        num_patches = (length[0] // patch_size[0]) * (length[1] // patch_size[1]) * (length[2] // patch_size[2]) # 总体有多少个 patch
        self.length = length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        ##### make sure an input of shape: (bs x y z c nt) #####

        x = x.flatten(4)                     ##### (bs x y z c*nt)
        x = x.permute(0, 4, 1, 2, 3)         ##### (bs c*nt x y z)
        x = self.proj(x)                     ##### (bs embed_dim x//px y//py z//pz)
        x = x.permute(0, 2, 3, 4, 1)         ##### (bs x//px y//py z//pz embed_dim)

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        ##### make sure an input of shape: (bs x//px y//py z//pz embed_dim) #####

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class Block(nn.Module):
    def __init__(
            self, nlayer, dim, patch_size, embed_dim, hidden_size_factor, num_blocks, in_chans, 
            drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, double_skip=True
        ):
        super().__init__()
        hidden_features = embed_dim * 4

        self.filter = AFNO(embed_dim, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(embed_dim, hidden_features, embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.double_skip = double_skip

    def forward(self, x):

        ##### must after patch_embed #####
        ##### input (bs x//px y//py z//pz embed_dim) #####

        residual = x

        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x # output = x + MLP(AFNO(x)) 

################################################################################################################################

class AFNO(nn.Module):
    def __init__(self, hidden_size, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, X, Y, Z, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = Z // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(X, Y, Z), dim=(1, 2, 3), norm="ortho")
        x = x.type(dtype)
        return x + bias

##################################################################################################################

class IAFNODiff(nn.Module):
    def __init__(
            self,
            dim, # (448, 448, 1) 单个大 patch 的维度，兼容当前 hdf5 数据集
            patch_size, # (8, 8, 1) 每个 patch 再被切成小块
            embed_dim, # 每个小 patch 被映射为多少维的特征 128
            num_blocks, # AFNO 在通道维度上划分的频率块数量
            cond_chans, # 输入通道数 7周SST + 一个mask = 8
            target_chans, # 输出通道数 未来15周SST
            ex_layer, # 
            nlayer, # layer 执行多少轮
            hidden_size_factor, # AFNO 每个频率块内部的通道扩张倍数
            dim_f, # 应等于 dim
            drop_rate=0., # 位置编码后的 dropout_rate
            sparsity_threshold=0.01, # AFNO频域系数的稀疏化阈值
            hard_thresholding_fraction=1.0, # 保留的频率模态比例
        ):
        super().__init__()

        if tuple(dim) != tuple(dim_f):
            raise ValueError(
                f"dim must be equal to dim_f without spatial padding, "
                f"but got dim={dim}, dim_f={dim_f}"
            )

        self.dim = dim
        self.dim_f = dim_f

        self.cond_chans = cond_chans
        self.target_chans = target_chans
        self.model_in_chans = cond_chans + target_chans
        self.in_chans = self.model_in_chans
        self.out_chans = self.target_chans

        self.ex_layer = ex_layer
        self.nlayer = nlayer
        self.patch_size = patch_size

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.patch_embed = PatchEmbed(dim, patch_size, embed_dim, self.in_chans)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim[0] // patch_size[0], dim[1] // patch_size[1], dim[2] // patch_size[2], embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = self.dim[0] // self.patch_size[0]
        self.w = self.dim[1] // self.patch_size[1]
        self.z = self.dim[2] // self.patch_size[2]

        self.blocks = nn.ModuleList([
            Block(
                nlayer, dim, patch_size, embed_dim, hidden_size_factor, num_blocks, self.model_in_chans
            )
            for i in range(self.ex_layer)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1]*self.patch_size[2], bias=False)

        time_embed_dim = 128
        hidden_chans = 2 * self.model_in_chans

        sinu_pos_emb = SinusoidalPosEmb(time_embed_dim, theta = 10000)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, hidden_chans * 2)
        )
        self.silu = nn.SiLU()
        self.rmsnorm1 = RMSNorm(2*self.in_chans)
        self.rmsnorm2 = RMSNorm(self.in_chans)

        self.upproj = nn.Conv3d(self.model_in_chans, hidden_chans, 3, padding=1)
        self.downproj = nn.Conv3d(hidden_chans, self.model_in_chans, 3, padding=1)

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        if self.ex_layer != 1 and self.nlayer == 1:
            for j in range(self.ex_layer):
                x = self.blocks[j](x)
        else:
            for i in range(self.nlayer):
                for j in range(self.ex_layer):
                    coef = 1/(self.nlayer * self.ex_layer)
                    x = x + self.blocks[j](x) * coef
        x = self.norm(x)

        return x

    def forward(self, x, time, condition):

        if condition is None:
            raise ValueError(
                "condition can't not be None"
            )

        if x.ndim != 5:
            raise ValueError(
                f"x must have [B, C, H, W, Z], but got {x.shape}"
            )

        if condition.ndim != 5:
            raise ValueError(
                f"condition must have shape [B, C, H, W, Z], "
                f"but got {condition.shape}"
            )

        if condition.shape[1] != self.cond_chans:
            raise ValueError(
                f"expected {self.cond_chans} condition channels, "
                f"but got {condition.shape[1]}"
            )

        if x.shape[0] != condition.shape[0]:
            raise ValueError("target and condition batch sizes do not match")

        if x.shape[2:] != condition.shape[2:]:
            raise ValueError(
                f"target spatial shape {x.shape[2:]} does not match "
                f"condition spatial shape {condition.shape[2:]}"
            )

        x = torch.cat((condition, x), dim=1)

        ##### time embedding process

        x = self.upproj(x)
        x = self.rmsnorm1(x)

        t = self.time_mlp(time)
        t = rearrange(t, 'b c -> b c 1 1 1')

        scale_shift = t.chunk(2, dim = 1)
        scale, shift = scale_shift

        x = x * (scale + 1) + shift
        x = self.silu(x)

        x = self.downproj(x)
        x = self.rmsnorm2(x)
        x = self.silu(x)
        
        x = rearrange(x, "bs c x y z -> bs x y z c")

        x = self.forward_features(x)
        x = self.head(x)

        x = rearrange(
            x,
            "b h w z (p1 p2 p3 c_out) -> b (h p1) (w p2) (z p3) c_out",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
            h=self.dim[0] // self.patch_size[0],
            w=self.dim[1] // self.patch_size[1],
            z=self.dim[2] // self.patch_size[2],
        )
        if (self.dim_f[0]!=self.dim[0]):
            x = x[:, :-1, :, :, :]
        if (self.dim_f[1]!=self.dim[1]):
            x = x[:, :, :-1, :, :]
        if (self.dim_f[2]!=self.dim[2]):
            x = x[:, :, :, :-1, :]

        x = rearrange(x, "bs x y z c -> bs c x y z")
        return x