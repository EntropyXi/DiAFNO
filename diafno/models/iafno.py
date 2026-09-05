# 用途：实现带时间条件的隐式自适应 Fourier 神经算子骨干。
# ---------------------------------------------------------------------------------------------
# Author: Yuchi Jiang
# LatestVersionDate: 07/27/2026 (specifically designed for diffusion)
# ---------------------------------------------------------------------------------------------

# Many thanks to all the authors of:
# Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B.: Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. arXiv preprint arXiv:2111.13587 (2021)

import math
from functools import partial

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath

# 全局 RNG 种子由训练器 set_random_seed(seed + rank) 统一管理，
# 不在此处硬编码，避免 import 时污染调用方的种子体系

################################################################################################################################

class SinusoidalPosEmb(nn.Module):
    # 用途：初始化正弦时间位置编码模块。
    # 参数：输入 dim（嵌入向量维度）、theta（频率基值，默认 10000）；输出 无（构造模块状态）。
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    # 用途：把标量时间/σ 值映射为 [sin, cos] 拼接的正弦嵌入。
    # 参数：输入 x（形状 [B] 的时间标量）；输出 形状 [B, dim] 的嵌入矩阵。
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
    # 用途：初始化 RMS 归一化层及其可学习缩放 g。
    # 参数：输入 dim（通道数，用于 sqrt(dim) 缩放补偿）；输出 无（构造模块状态）。
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    # 用途：沿通道 L2 归一化并乘 sqrt(dim)*g，等效标准 RMSNorm。
    # 参数：输入 x（任意维度特征张量）；输出 归一化后的同形张量。
    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale # normalize 是 L2 norm 需要补一个*sqrt(dim)转为标准RMSNorm

# 用途：判断对象是否非 None 的工具函数。
# 参数：输入 x（任意对象）；输出 布尔值。
def exists(x):
    return x is not None

# 用途：val 为 None 时回退到默认值的工具函数。
# 参数：输入 val（可能为 None 的值）、d（默认值或无参工厂）；输出 val 或 d 的结果。
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 把 3D 空间场切成 3D patch，并把每个 patch 映射成一个 embedding vector
class PatchEmbed(nn.Module):
    # 用途：初始化 3D patch 嵌入（Conv3d 按步长切块并升维）。
    # 参数：输入 length（输入空间尺寸 [X,Y,Z]）、patch_size（每块尺寸）、embed_dim（每块嵌入维度）、in_chans（输入通道数）；输出 无（构造模块状态）。
    def __init__(self, length, patch_size, embed_dim, in_chans):              #####   Length & Patch_size must be 3 dims   #####
        super().__init__()
        num_patches = (length[0] // patch_size[0]) * (length[1] // patch_size[1]) * (length[2] // patch_size[2]) # 总体有多少个 patch
        self.length = length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    # 用途：把条件+目标拼接后的通道展平为 Conv3d 输入并切块嵌入。
    # 参数：输入 x（[B,C,H,W,Z] 拼接场）；输出 形状 [B,X',Y',Z',embed_dim] 的 token 特征。
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
    # 用途：初始化两层 MLP（GELU 激活 + dropout）。
    # 参数：输入 in_features/hidden_features/out_features（输入/隐层/输出宽度）、act_layer（激活类）、drop（dropout 概率）；输出 无（构造模块状态）。
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # 用途：对 token 特征做 fc1-激活-fc2 的 MLP 变换。
    # 参数：输入 x（[B,X,Y,Z,C] 特征）；输出 同形变换后特征。
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
    # 用途：初始化 AFNO 滤波 + MLP 的残差块单元。
    # 参数：输入 embed_dim（token 维度）、hidden_size_factor（AFNO 通道扩张倍数）、num_blocks（频率分块数）、drop_path（随机深度率）、double_skip（双残差开关），其余为兼容旧签名的占位；输出 无（构造模块状态）。
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

    # 用途：前向执行 x = x + MLP(AFNO(norm(x)))（double_skip 时为双残差）。
    # 参数：输入 x（patch token 特征 [B,X,Y,Z,C]）；输出 变换后的同形特征。
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
    # 用途：初始化 AFNO 的复数域块对角两层权重（w1/b1/w2/b2）与稀疏化参数。
    # 参数：输入 hidden_size（通道数）、hidden_size_factor（块内扩张倍数）、num_blocks（频率块数）、sparsity_threshold（softshrink 阈值）、hard_thresholding_fraction（保留高频比例）；输出 无（构造模块状态）。
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

    # 用途：3D rFFT -> 分块两层 MLP -> softshrink 稀疏化 -> irFFT，并加输入残差。
    # 参数：输入 x（[B,X,Y,Z,C] token 特征）；输出 频域全局混合后的同形特征（含 bias 残差）。
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
    # 用途：构建 IAFNO 去噪网络：patch 嵌入、位置编码、外层 Block 组、σ 嵌入 FiLM 注入与升降卷积。
    # 参数：输入 dim（空间尺寸 [448,448,1]）、patch_size（[8,8,1]）、embed_dim（嵌入维度）、num_blocks（频率块数）、cond_chans（条件通道：7 日 SST+mask=8）、target_chans（目标通道 15）、ex_layer（外层块数）、nlayer（隐式迭代轮数）、hidden_size_factor（扩张倍数）、dim_f（须等于 dim）、drop_rate（位置编码 dropout）；输出 无（构造模块状态）。
    def __init__(
            self,
            dim, # (448, 448, 1) 单个大 patch 的维度，兼容当前 hdf5 数据集
            patch_size, # (8, 8, 1) 每个 patch 再被切成小块
            embed_dim, # 每个小 patch 被映射为多少维的特征 128
            num_blocks, # AFNO 在通道维度上划分的频率块数量
            cond_chans, # 输入通道数 7天SST + 一个mask = 8
            target_chans, # 输出通道数 未来15天SST
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

    # 用途：跑主干：nlayer 轮隐式迭代 × ex_layer 个块（残差系数 1/(nlayer*ex_layer)），末尾 LayerNorm。
    # 参数：输入 x（patch token 特征）；输出 归一化后的 token 特征。
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

    # 用途：去噪前向：拼接条件与加噪目标，σ 嵌入经 FiLM scale/shift 调制后过主干与 head 还原为像素场。
    # 参数：输入 x（加噪目标 [B,target_chans,H,W,Z]）、time（c_noise 编码 [B]）、condition（条件场 [B,cond_chans,H,W,Z]）；输出 形状同目标的网络输出 F_θ（EDM 预条件前的原始输出）。
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
