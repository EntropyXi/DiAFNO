# DiAFNO 论文原版（main 分支）与 OSTIA 日尺度 SST 实现的差异分析

> 对比对象
>
> - **main 分支（论文原版）**：`IAFNO.py` / `diffusion.py` / `trainer.py` / `utilities3.py`，对应论文 *Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence*（Jiang et al., arXiv:2512.12628）。任务：三维湍流（64×66×32 网格、3 个速度分量）的**自回归一步预测**。
> - **OSTIA_SST 分支（本任务）**：`diafno/` + `deterministic_iafno/` + 各入口脚本。任务：**连续 7 日 SST 输入 → 随后 15 日 SST 直接预测**（centered diffusion）。
>
> 本文按 dataset/dataloader、模型输入输出、IAFNO 网络、diffusion 目标与 loss、训练工程、评估协议六个维度逐项对比，并给出每一项差异的实现原理与依据（多数都能在本项目 Phase 0–9 实验记录中找到对应教训）。

---

## 0. 总览

| 维度 | main（论文原版） | OSTIA_SST（本任务） |
|---|---|---|
| 任务 | 3D 湍流自回归（x(t) → x(t+1)） | 7 日 SST → 15 日 SST 直接预测 |
| 数据 | 单文件 `np.load`，20 条轨迹 × 200 窗口 | HDF5 1,126,100 行（11261 日 × 100 空间块） |
| 划分 | `random_split` 80/20（重叠窗口随机拆） | 时间顺序 70/20/10（防泄漏） |
| 归一化 | 逐通道 min-max + **数据全量 std 作为 sigma** | train-only z-score + train-only lead 统计 |
| 条件方式 | self-conditioning（上一帧与噪声目标拼接） | 显式 7 日条件（含 mask）与目标在主干内拼接 |
| 输出 | 3 通道同形状速度场 | 15 通道未来 SST（创新项空间） |
| sigma_data | 0.5（trainer 用数据 std 覆盖） | **1.0**（fail-closed 强制） |
| S_churn | **80**（默认开） | **0**（消融后锁定） |
| loss | 全网格 MSE（无 mask）× λ(σ) | **海洋掩码** per-sample MSE × λ(σ) |
| 训练 | 单卡 flat 脚本，无 checkpoint 语义 | DDP + sidecar 语义 + fail-closed resume |
| 评估 | LpLoss（相对 L2）+ 湍流统计量 | 掩码 RMSE/MAE/skill + bootstrap CI + CRPS/coverage |

---

## 1. 任务与数据

### main
- 三维湍流速度场（64×66×32，3 分量），`data[0:20,...,0:3]`，窗口 `[x(t), x(t+1)]`（InferenceWidth=1、InitialInterval=1，即一步自回归）。
- 数据是内存张量：`np.load` 后 `torch.stack` 成 `TensorDataset`。
- **随机 80/20 划分**：滑动窗口高度重叠，随机拆会把相邻时刻同时放进 train/test，测试集与训练集时间相关，泛化结果被高估。
- 归一化：逐通道 min-max 缩放到 [0,1]，再**用训练集整体 std 作为 sigma**（保存到 `max_min_sigma info` 文件复用）。

### OSTIA
- 全球日尺度 SST（448×448×1 空间块，100 块/日），样本 = 同一空间块的 22 个连续自然日（7 输入 + 15 输出）。
- **时间顺序划分** 70/20/10（按日索引），val/test 永远不与 train 日期重叠——这是 Phase 0 定的硬规则（`OSTIADailyDataset.split_ranges`）。
- 归一化：**train split 的 z-score**（`sst_mean/sst_std` 从 HDF5 attrs 或 train 样本估计），val/test 不参与任何统计。
- 无效像素（mask 位、非有限、物理范围外）用 train 均值填充进张量，但由 mask 从 loss/指标中剔除。

**差异依据**：论文场景是封闭数值实验（数据生成过程同分布、窗口重采样可接受）；地球物理预报必须防时间泄漏（实验记录 Phase 0、Phase 6 的诊断直接由「打乱条件结果不变」暴露了条件利用问题）。

---

## 2. Dataset 与 DataLoader

### main
- `TensorDataset` + 标准 `DataLoader(batch_size=4, shuffle=True)`；数据全程在内存；无 worker/预取调优；无采样器（单卡）。

### OSTIA
- `OSTIADailyDataset.__getitems__`：按 sequence 分组，同 sequence 的样本合并读取；HDF5 行按 `day × samples_per_day + spatial_index` 定位，**连续空间块成段读取**（对齐 HDF5 chunk，避免每样本一次随机 IO）——来自 Phase 5 的 `/data2` 本地盘 + chunk-aware 读优化。
- `DistributedSpatialBlockSampler`：把样本按「日序列 × 空间块」分块，每 epoch 随机排列块、按 rank 切分、空间起始偏移随 epoch 轮转——保证 DDP 下各 rank 样本不重叠、同一天的空间块尽量同 batch。
- `DataLoader`：`persistent_workers`、`prefetch_factor`、`pin_memory`、`worker_init_fn` 种子。
- 每个样本带 metadata（sequence/spatial index、输入/目标起止时间）——bootstrap 与诊断要用。

**差异依据**：1.1M 行 HDF5 的随机单样本读取会打满磁盘队列（Phase 5 实测），chunk-aware 分组读取把吞吐拉回训练预算内；时间块 bootstrap 需要每个样本的初始化时间（Phase 7）。

---

## 3. 模型输入输出

### main
- 输入 = 上一帧速度场 x（3 通道），输出 = 下一帧同形状场。
- **self-conditioning**：`self_condition=True` 时把「上次采样的输出/加噪目标」与当前噪声目标在通道维拼接（`in_chans × 2`），实现类 RePaint 的迭代细化；训练时随机 50% dropout 的 self-cond 分支在代码里被注释掉了（实际总是用真实 self_cond）。
- 张量布局 4D：`(B, C, X, Y, Z)`。

### OSTIA
- 条件 = 7 日 SST + 最后输入日 mask（8 通道），目标 = 15 日未来 SST（15 通道）；布局 5D：`(B, C, H, W, 1)`（Z=1，为复用 3D 卷积）。
- **条件与噪声目标在 IAFNODiff.forward 内部拼接**（`torch.cat((condition, x), dim=1)`），没有 self-conditioning 分支。
- 关键语义：训练目标先做三级变换 `z = ((y − anchor − μ) − m)/s`（残差 → centered innovation → lead 标准化），模型学 `p(z|c)`；推理时 wrapper 的 `sample()` 输出 normalized residual `μ + m + s·ẑ`，**anchor 由 evaluator 只加一次**。
- 确定性路径复用同一主干：`DeterministicIAFNO` 输入零目标 + 固定时间 0.0，直接回归（绕过 EDM 预条件化）。

**差异依据**：SST 的绝对动态范围与海区气候差会吞掉容量（Phase 3 教训）；anchor 只加一次、mean 只加一次是 Phase 2 计划的代数不变量（对应 wrapper 单测锁定）。

---

## 4. IAFNO 网络

两侧主干结构几乎同源（PatchEmbed → 位置编码 → 时间 MLP → RMSNorm → AFNO Block × N → head），但细节差异显著：

| 项 | main | OSTIA |
|---|---|---|
| `in_chans` | `in_chans × (2 if self_condition)` | `cond_chans + target_chans`（8+15） |
| 时间嵌入 | `SinusoidalPosEmb(dim=in_chans)`，`time_mlp: in→4in→4in` | `SinusoidalPosEmb(dim=128)`，`time_mlp: 128→512→2×hidden`（`hidden=2×in_chans`），scale/shift 调制 |
| 头 | `head: embed_dim → out_chans×p1p2p3` | 同 |
| 空间填充 | dim(64,66,32) vs dim_f(64,65,32)：**在 patch 前补零**再切 | dim == dim_f（448 整除 patch），只保留防御性裁剪分支 |
| 位置编码 | 学习式 `pos_embed`（零初始化） | 同 |
| 层 | `embed_dim=180, ex=4, nlayer=4`（implicit 循环） | `embed_dim=128, ex=4, nlayer=2`（Phase 1 冻结配置） |
| DropPath | `drop_path_rate` 随层线性 | 关闭（drop_rate=0） |
| 设备 | `Block(...).cuda()` **硬编码** | `model.to(device)` 统一管理 |
| 归一化/激活 | RMSNorm + SiLU + GELU（MLP） | 同 |
| 随机种子 | import 时 `torch.manual_seed(123)` | 由 trainer `set_random_seed(seed+rank)` 统一管理（不污染调用方） |

主干数学本质相同（AFNO 复数频域 MLP + softshrink 稀疏化），差异主要是**条件通道组织**（self-cond vs 显式条件）与**工程卫生**（硬编码 cuda、import 种子、padding 位置）。

---

## 5. Diffusion 目标与 loss

### main
- 目标 = **未来帧本身**（min-max 后），`sigma_data=0.5`（trainer 用数据 std 覆盖）。
- 噪声：`ln σ ~ N(P_mean=-1.2, P_std=1.2)`，`σ ∈ [0.002, 80]`。
- 预条件化与 OSTIA 相同（EDM Table 1 的 c_skip/c_out/c_in/c_noise）。
- **loss = 全网格逐点 MSE 的 per-sample 均值 × λ(σ)，batch 平均**：

```text
L = E_σ [ λ(σ) · mean_{所有网格点} ( D_θ(z_tilde; σ, c) − z )² ]
```

- 没有 mask（数值模拟场处处有效）；没有 anchor/创新项（自回归一步预测天然是「变化」目标）。
- **S_churn = 80 默认开启**（高注入噪声的随机采样器），另带 `clamp` 与 DPM++ 采样器变体。

### OSTIA
- 目标 = **train-only 标准化创新项 z**（见 §3），`sigma_data=1.0`（配置层与 build_model 双重 fail-closed，拒绝其他值）。
- 噪声分布、预条件化完全同源。
- **loss = 掩码 per-sample MSE × λ(σ)**：

```text
L_s = λ(σ) · mean_{Ω_s} ( D_θ − z )²        Ω_s = 该样本未来 15 日有效海洋像素
L   = mean_s L_s
```

- 实现约束：`e`、`z` 的减法/除法在 **fp32、autocast 之外**；冻结均值 `no_grad` 且不进优化器；梯度裁剪 1.0；AMP overflow skip 不推进 optimizer/scheduler（计数入 checkpoint）。
- `S_churn = 0`（epoch 5 消融锁定；实验记录 Phase 3 曾因 churn 导致 10.4 K → 3.6 K 的惨案）。
- 确定性路径的 loss 是另一套：`globally_normalized_masked_mse`（DDP 下按全局有效像素数归一化，保证各 rank 平均梯度 = 全局掩码 MSE 梯度）。

**差异依据**：
1. mask 是必须的——海洋预报只有有效海域像素有真值（Phase 0 口径）；
2. centered 目标 + sigma_data=1 让 EDM 的噪声区间恢复标准含义（Phase 8 设计核心），旧残差路径的 0.15 是硬凑；
3. churn=0 与 16 steps 是 Phase 3/6 的采样教训的直接产物。

---

## 6. 训练工程

| 项 | main | OSTIA |
|---|---|---|
| 并行 | 单卡 flat 脚本 | `torch.distributed.run` DDP（1/2 卡，有效 batch 32） |
| AMP | `GradScaler` 包 loss.backward | GradScaler + **overflow skip 计数**（不推进 scheduler） |
| 优化器 | （脚本内常规） | AdamW lr 2e-4→1e-6 cosine、wd 1e-4、clip 1.0 |
| checkpoint | 无（无 resume 语义） | schema 4 自包含 checkpoint + sidecar（immutable 语义、mean/stats 双 SHA）+ fail-closed resume + 每 rank 随机状态 |
| 数据语义保护 | 无 | 冻结 mean 身份 SHA、stats provenance、`split=train` 强制校验 |
| 观测 | 训练/测试 loss 打印 | loss/梯度/跳过步曲线 npz+PNG、epoch 摘要、launch manifest、watcher |

**差异依据**：main 是论文复现脚本，工程语义从简；OSTIA 的所有 checkpoint/resume 机制都来自 Phase 5 的 resume 语义漂移事故（p_mean -3→-1.2 导致「假进步」）——记录里明确为三条规则：sidecar 语义、fail-closed、固定协议选择。

---

## 7. 评估与验证协议

| 项 | main | OSTIA |
|---|---|---|
| 点指标 | `LpLoss`（相对 L2），加湍流统计量（能谱、RMS、Reynolds 应力） | 掩码 RMSE/MAE/bias/corr/std_ratio（overall + 15 lead）+ MSE skill vs persistence |
| 置信区间 | 无 | **paired temporal block bootstrap**（22 日块、10000 次、95% CI） |
| 概率指标 | 无 | **CRPS / coverage / spread-skill / 成员收敛曲线** |
| 协议锁定 | 无 | 固定 val-200 seed 123；采样协议消融后锁定 16 steps × 4 members × churn 0；test-1000 只跑一次 |
| 选择 | （未涉及） | `best_val_mean_rmse.pth`（点 RMSE 最优，按锁定协议） |

**差异依据**：SST 场景有强时间相关，像素级独立假设会虚假收窄 CI（Phase 7 引入块 bootstrap）；diffusion 的判决必须用概率指标（Phase 10/15 的 CRPS 结论）。

---

## 8. 为什么这些差异是「必要」而不是「风格」

1. **防泄漏**：地球物理数据的时间顺序划分 + train-only 统计是硬需求；论文的封闭数值实验可以随机拆。
2. **目标尺度**：绝对 SST（3.6 K）→ 残差（1.37 K）→ centered innovation（1.18 K 点 / CRPS 0.39）是三次目标空间收敛；每次都是上一阶段失败的直接修正。
3. **mask**：海洋任务没有 mask 等于让网络背陆地目标。
4. **条件利用**：Phase 6 消融证明旧 diffusion 几乎不用条件，centered 设计把「均值」与「扰动」职责分离，条件利用率才可验证。
5. **分布建模定位**：论文的 diffusion 目标是「更准的场重建」（自回归框架），本任务的 diffusion 目标是「围绕确定性均值的校准分布」——所以判据从 RMSE 换成 CRPS/coverage。
6. **工程语义**：服务器共享 + 断点 + 多卡要求 checkpoint 语义、AMP skip 计数与 fail-closed resume；论文脚本不需要。

---

## 9. 参考位置

- main 分支：`git show main:IAFNO.py`、`main:diffusion.py`、`main:trainer.py`
- OSTIA 实现：`diafno/models/iafno.py`、`diafno/models/diffusion.py`、`diafno/data/ostia.py`、`diafno/training/*`、`deterministic_iafno/centered_diffusion.py`、`deterministic_iafno/losses.py`
- 实验教训出处：`D:\Notes in Datahub' s Learning\深度学习笔记\神经算子类\DiAFNO OSTIA日尺度SST预测实验记录.md`（Phase 0–17）
