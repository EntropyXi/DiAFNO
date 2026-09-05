# 分支 A' 计划 v2：绝对模型权重移植 → 残差目标微调（DiAFNO OSTIA）

状态：**v2.1，已按双高精度审查修订 + best_model.pth 记录机制，待主训练结束执行**
评审人：A（EDM/训练动力学，subagent 0cc3f709）、B（工程/执行路径，subagent ff6c9540）
v1→v2 变更摘要见文末 §9。

## 0. 目标与验收（v2 重写）

**目标**：绝对 35-epoch checkpoint 权重移植到残差模式微调，使固定 200 索引验证达到
- **硬门槛（epoch 3，唯一判定依据）**：day1 RMSE ≤ 0.5 K **且** overall ≤ 1.4 K（= persistence 1.19 K × 1.2）；
- **损失仅作健康参考**：在新 σ 分布 (σ_data=1.0, P_mean=−3.0) 下，残差逐 lead 收敛下限为 0.30–0.62、lead 平均 ≈0.45–0.55（MC 实测）——**不是 0.13–0.15**（那是旧 P_mean=−1.2 下的值）。健康判据：损失有限、单调下降、无 NaN/inf 梯度；参考值 epoch1 <0.8、epoch3 ≈0.55±0.1。
- 不达硬门槛 → 切分支 B（§6）。

**证据基础（同 200 索引，含 σ_data 标注）**：
| 配置 | overall | day1 | 注 |
|---|---|---|---|
| 绝对 churn=80/16步/ens1 | 10.41 K | 9.96 K | σ_data=1.0 |
| 绝对 churn=0/32步/ens8 | 3.64 K | 3.20 K | σ_data=1.0 |
| 残差烟测 1 epoch（churn=0/ens4） | 1.50 K | 0.83 K | **σ_data=0.15**（训练默认已是 0.15/residual，5035fef 引入） |
| persistence | 1.19 K | 0.25 K | — |

**因果假设（自洽、待 fixed-σ probe 证实，见 §4）**：D_r = D_abs(r_σ+a,σ,cond) − a 严格成立（点态、任意 σ、含 c_skip/c_out 耦合——评审 A 复核为"输入平移+输出减 a"的仿射重参数化）；绝对权重含全部特征结构（corr 0.92、方差校准 8.81≈8.62）。绝对模型 day1=3.2 K 的机制假设：条件均值 m 在 σ≈0.3–1 的最优网络输出中振幅大（可学），但**采样器最后几步（σ<0.01）在网络从未充分训练的外推区评估**（P(σ<0.01)=0.23%@−1.2 → 全训练 ~1500 样本；σ<0.002 仅 ~10 样本），外推爆炸（pred_min=256.8 K 为佐证）。备择假设：采样路径离散化问题。**判别实验 = fixed-σ probe（§4）。**

## 1. 代码改动清单（~40 行，全部本地改→提交→push；主训练退出后才在服务器 git pull）

### 1.1 `diafno/models/config.py` — 3 个新字段（**默认值保持旧值：80.0 / 0.002 / −1.2，作为永久不变量保护旧 checkpoint 可复现性**）
```python
sigma_max: float = 80.0
sigma_min: float = 0.002
p_mean: float = -1.2
```
`build_model()` 透传三者给 `ElucidatedDiffusion`（构造器已支持，`diffusion.py:44-50`）。`to_checkpoint/from_checkpoint` 无需改（asdict + field_names 过滤自动携带；旧 checkpoint 缺字段回落默认旧值 → 旧绝对 checkpoint 验证行为不变）。微调 checkpoint 会把 1.0/0.0005/−3.0 存入 config → **验证器自动使用，无需再传参**（评审 A P2-4）。
注：S_churn/S_noise/rho 不在 config 中，跟随代码默认值（当前 S_churn=0）；复现旧 10.4 K 需显式 `--s-churn 80`（评审 B P2-1）。

### 1.2 `diafno/training/config.py` — CLI 透传
`build_parser()` 增 `--sigma-max/--sigma-min/--p-mean`（type=float）；`training_config_from_args()` 写入 `config.model.<field>`（仿照现有 `--sigma-data` L115-116 模式）。
**警告写入文档**：`OSTIATrainingConfig` 默认 factory 已是 `OSTIAModelConfig(sigma_data=0.15, target_mode="residual")`（`training/config.py:14-19`）——**微调命令必须显式 `--sigma-data 1.0`**，漏传会静默用 0.15 导致权重语义错配。

### 1.3 `diafno/training/trainer.py` + `config.py` — `--init-from`
- 新字段 `init_from: Optional[str] = None` + CLI `--init-from`；与 `--resume` 互斥（parser 直接报错）。
- `setup()` 中 `_build_training_components()` 后、`_resume_training()` 前，当 `resume_path is None and init_from is not None` 执行 `_init_from_checkpoint()`：
  1. `torch.load(init_from, map_location="cpu", weights_only=False)`；
  2. 源配置 = `OSTIAModelConfig.from_checkpoint(checkpoint["config"])`；校验除 `target_mode` 外全部架构字段与当前一致；**额外校验源 `sigma_data` == 当前 `sigma_data`**（防权重语义错配，评审 B P1-2）；
  3. `CheckpointManager.unwrap_model(model).load_state_dict(state_dict, strict=True)`；失败时包装为含"源路径 + 源/目标字段对照表"的 RuntimeError；
  4. **不加载** optimizer/scheduler/scaler/random_states → `start_epoch=0`、`global_step=0`、全新 AdamW + cosine（T_max=35×975=34125，accum=1 双卡）；
  5. 加载完成后 `self.runtime.barrier()`（评审 B P2-5）；
  6. 主进程打印移植日志（源路径、target_mode 源/目标、参数张量数）。
- 权重移植不改变计算图 → `find_unused_parameters=False` 继续成立（评审 B 已核实）。

### 1.4 `diafno/evaluation/validator.py` + `config.py` — fixed-σ probe 模式（评审 A P1-1 的判别实验）
- 新 CLI `--probe-sigma`（float，可多次或逗号分隔）；设置后 `prediction_mode="probe"`：
  - absolute target_mode：`noised = target + σ·randn_like(target)`（按 member seed 生成），`D = model.preconditioned_network_forward(noised, σ, condition)`；
  - residual target_mode：`noised = (target − anchor) + σ·randn_like(...)`，预测 = D + anchor；
  - 成员平均后走现有指标路径（mask 一致、反变换一致）。
- 目的：σ=0.002 的 D 直接近似条件均值 E[x|cond]，绕过采样器，测量"病灶到底在哪"。

### 1.5 明确不改
`_valid_ocean`/数据集/采样器/DDP 结构/EDM preconditioning 公式/metrics 公式 **不改**；物理钳位（pred clip）不做（只在验收时观察 metrics 已输出的 pred min/max）；锚点过渡像素（day7 无效而 day8+ 有效）**实测为 0**（评审 A P2-1：3 天×6 窗口抽样 46.5M 像素），不加代码、仅记录。

### 1.6 新文件 `scripts/finetune_epoch_watcher.py` — best_model.pth 离线监视器（v2.1 新增）
设计原则：**训练进程零改动、零占用训练卡**；监视器作为独立进程跑在服务器上，验证走 GPU 2/3（batch-1 ≈1 GB，与现有任务共存已实测）。
循环逻辑：
1. 轮询微调日志 `epoch=N train_loss=` 行 / `latest.pth` mtime，检测新 epoch 完成；
2. `sleep 5` 后快照 `cp latest.pth /tmp/ostia_ft_snapshots/epochN.pth`（防 torch.save 非原子写竞态，评审 B P1-3）；
3. GPU 2/3 上跑 `validate_ostia.py`（200 固定索引、16 步、`--s-churn 0 --ensemble-members 4`，输出 `validation_ft_epochN.json`）；
4. 解析 overall RMSE（主指标）与 day1 RMSE（次级，同分取小），连同 train_loss 逐行追加写 `/tmp/ostia_ft_logs/epoch_metrics.jsonl`；
5. **选优**：若 overall RMSE 优于当前最优 → 将快照原子复制（`cp` 到临时名 + `mv`）为 `experiments/ostia_7day_to15day_residual_ft/best_model.pth`（trainer 永不写该文件名，无冲突）；
6. 训练进程退出后：终检一次并写汇总，退出。
附带效果：epoch 1/2/3 验收门数据由它自动采集（§4），人工只需做分支判定。

## 2. 微调超参数（全部显式经 CLI）

| 参数 | 值 | 依据 |
|---|---|---|
| --target-mode | residual | 目标重锚定 |
| --sigma-data | **1.0（必须显式传！默认已是 0.15）** | 移植权重语义兼容（评审 A/B P0-2） |
| --sigma-max | 1.0 | 训练分布 P(σ>1)=0.62%@−3.0，起采 1.0 覆盖充分 |
| --sigma-min | **0.0005**（v2 由 0.002 改） | 末步后验噪声 ≈0.005 K；与分支 B 对齐（评审 A P2-2） |
| --p-mean | -3.0 | σ 中位数 0.05 ≈ 残差信息尺度；P(σ<0.01)=9.2% → 每 epoch ~2870 样本覆盖采样器末段（评审 A 复核 50.14%/0.62%/0.37%/0.0498 全对） |
| --learning-rate | **1e-4**（v2 由 5e-5 改） | 头原始输出需缩 ~10×，5e-5 需 1–2 epoch 偏紧（评审 A P1-2）；无 warm-up/不冻结 |
| --max-grad-norm | 1.0；clip 率>30% 且损失不降 → 2.0 | 烟测 17% clip 属预期 |
| batch/accum/seed/workers | 16 / 1 / 123 / 4 | 不动 |
| --num-epochs | 35（1/2/3 epoch 高频验收，可随时止损） | 验收期 ~1.5–2h |
| --output-dir | experiments/ostia_7day_to15day_residual_ft | 与冒烟目录隔离（评审 B P1-4） |

## 3. 启动时序（严格防干扰，v2 修订）

0. **前置**（主训练未结束也可做）：GPU 2/3 上跑 **绝对 checkpoint 的 fixed-σ probe 基线**（σ∈{0.002, 0.05}，~10 分钟）——为 epoch-3 probe 对比建档；
1. 所有代码改动**只在本地**完成 → commit → push（本地=origin=8581bca 已同步，无未推送提交）；
2. 主训练退出三重确认：`pgrep -f trainer_ostia.py` 为空 + `epoch_035.pth` 落盘 + GPU 0/1 显存回落（监视器 pwsh-2 已在执行此逻辑；评审 B P2-4）；
3. 服务器 `git status --short` 复核（当前仅 3 个未跟踪验证 JSON，不影响快进 pull）→ `git pull`；
4. GPU 0 冒烟：`--samples-per-epoch 1600`（= 100 optimizer 步；batch16 单卡时 sampler 的最小可行是 16，评审 B P2-2）→ `--output-dir experiments/ostia_residual_ft_smoke` → 检查：移植日志正常、损失有限且下降、无 NaN/inf 梯度；
5. GPU 0/1 启动 35 epoch 微调（nohup，日志 `/tmp/ostia_ft_logs/ft.log`；环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，评审 B P1-4）；
6. **启动 epoch 监视器（§1.6）**：自动完成"检测 epoch 完成 → sleep 5 → 快照 latest.pth → GPU 2/3 验证（16 步/churn0/ens4、200 固定索引）→ 记录 train_loss/day1/overall → overall RMSE 最优者原子复制为 `best_model.pth`"；验收门数据由它自动采集。

启动命令模板：
```bash
cd /data2/user/zzx/exam_preprocessed/DiAFNO && git pull
mkdir -p /tmp/ostia_ft_logs /tmp/ostia_ft_snapshots
# 冒烟
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 nohup \
python -u trainer_ostia.py --output-dir experiments/ostia_residual_ft_smoke \
  --target-mode residual --sigma-data 1.0 --sigma-max 1.0 --sigma-min 0.0005 --p-mean -3.0 \
  --init-from experiments/ostia_7day_to15day/latest.pth \
  --learning-rate 1e-4 --samples-per-epoch 1600 --batch-per-gpu 16 --gradient-accumulation 1 \
  --num-epochs 1 --checkpoint-interval 1 > /tmp/ostia_ft_logs/smoke.log 2>&1 &
# 正式微调（冒烟通过后）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 nohup \
python -u -m torch.distributed.run --standalone --nproc_per_node=2 trainer_ostia.py \
  --output-dir experiments/ostia_7day_to15day_residual_ft \
  --target-mode residual --sigma-data 1.0 --sigma-max 1.0 --sigma-min 0.0005 --p-mean -3.0 \
  --init-from experiments/ostia_7day_to15day/latest.pth \
  --learning-rate 1e-4 --batch-per-gpu 16 --gradient-accumulation 1 --num-workers 4 \
  --checkpoint-interval 5 > /tmp/ostia_ft_logs/ft.log 2>&1 &
```

## 4. 验收与实验矩阵（v2.1 修订；验证由 §1.6 监视器自动执行、验证前一律快照 checkpoint）

| 时机 | 实验 | 判读（v2） |
|---|---|---|
| 前置 | 绝对 checkpoint fixed-σ probe（σ=0.002, 0.05） | 建档基线（预期 σ=0.002 RMSE ≈3 K 量级） |
| 冒烟 100 步 | 损失曲线 + 梯度 | 健康=有限、下降、无 NaN/inf；不设绝对数值门槛 |
| epoch 1 | 验证 16步/churn0/ens4 | 信息性观察（头重标定可能未完成，day1 可虚高） |
| epoch 2 | 同上 + persistence | day1 逼近 0.5 K 的趋势 |
| epoch 3 | **硬门槛**：day1≤0.5 K 且 overall≤1.4 K；+ fixed-σ probe（σ=0.002） | 过→继续；不过→分支 B。probe 判读：微调后 σ=0.002 RMSE 从 ~3 K 降到 <0.5 K → "σ 外推区未训练"病灶证实；不变 → 问题在采样路径，另查（评审 A P1-1） |
| 模型 overall **低于 persistence 之后** | zero_sst 消融（+ 同测 train-split） | **判读修正**：zero_sst 后 RMSE 回落到 ≈persistence 水平 = 条件被真实利用；"指标几乎不变"在模型尚未超过 persistence 前是预期行为、不是忽略条件（评审 A P1-3） |
| 每 5 epoch | checkpoint 常规保存 | 横向对比收敛轨迹 |

**best_model.pth 选取规则（v2.1）**：主指标 overall RMSE（K），次级 day1 RMSE（同分取小）；每 epoch 验证后若优于当前最优，快照原子复制（tmp+mv）为 `experiments/ostia_7day_to15day_residual_ft/best_model.pth`；逐 epoch 指标汇总写 `/tmp/ostia_ft_logs/epoch_metrics.jsonl`。

## 5. 风险与缓解（v2 修订）

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 输出头重标定（原始输出 ~10× 收缩，非 400×） | 低 | lr 1e-4；epoch 1/2 信息性观察不设硬门槛 |
| R2 σ_data=1.0 与残差尺度的 preconditioning 失配 | 中 | 采样数学自洽；init-from 处加 σ_data 一致性断言；若采样质量差 → 二阶段微调降 σ_data（彼时权重已近残差语义） |
| R3 checkpoint/字段兼容 | 低 | 默认值不变式 + from_checkpoint 机制；strict load 报错包装 |
| R4 残差梯度方差大、clip 频繁 | 中 | 观察 clip 率；>30% 且不降损 → max_grad_norm 2.0 |
| R5 与主训练互相干扰 | 高 | 服务器文件在主训练退出前零改动；三重退出确认；验证走 GPU 2/3 |
| R6 latest.pth 读竞态 | 中 | 快照后验证（评审 B P1-3） |
| R7 微调不收敛/验收误判 | 中 | 硬门槛只认验证 RMSE（与 σ 分布无关）；损失参考值按新分布重标定 0.45–0.55 |
| R8 训练默认值陷阱（σ_data=0.15） | 高 | 命令显式 `--sigma-data 1.0`；init-from 加一致性校验 |

## 6. 分支 B 回退方案（与 A' 共享改动，参数对齐）

从零训残差：同 1.1–1.2 改动；`--sigma-data 0.15 --sigma-max 1.0 --sigma-min 0.0005 --p-mean -3.0`，无 `--init-from`；预期 5–10 epoch 收敛；烟测（σ_data=0.15 实际）已证 pipeline 可行。σ_min=0.0005 与 A' 对齐减少变量。

## 7. 成功定义

- 硬门槛（day1≤0.5 K、overall≤1.4 K）达成并持续；lead 5–15 逐 lead ≤ persistence 且长 lead 反超（persistence lead5–15 = 0.95→1.58 K）；
- fixed-σ probe 证实条件均值修复（σ=0.002 RMSE <0.5 K）；zero_sst 消融显示回落到 persistence 水平（条件被利用）；
- `best_model.pth` 落盘且与验收结果一致（由 §1.6 监视器自动维护）；全部改动提交 git、命令/seed/参数全记录可复现。

## 8. 待执行依赖清单

- [ ] 主训练退出（监视器 pwsh-2 通知）
- [ ] 代码改动 1.1–1.4 落地 + commit + push（等主训练退出后才能在服务器 pull）
- [ ] 前置 probe 基线（可在等待期用 GPU 2/3 先行）
- [ ] 冒烟 → 启动微调 + epoch 监视器（best_model.pth）→ epoch 1/2/3 验收

## 9. v1→v2 变更摘要

1. **验收损失线重标定**（A P0-1 / B P1-1）：删除 loss≤0.20 硬门槛；硬门槛只认验证 RMSE；损失参考值改为新分布下 0.45–0.55；
2. **σ_data 事实修正**（A P0-2 / B P1-2）：证据表标注烟测 σ_data=0.15；命令显式 `--sigma-data 1.0`；init-from 增加 σ_data 一致性校验；
3. **新增 fixed-σ probe 判别实验**（A P1-1）与 validator 支持（§1.4、§4）；
4. **lr 5e-5 → 1e-4**（A P1-2）；σ_min 0.002 → 0.0005（A P2-2）；
5. **zero_sst 判读逻辑修正**（A P1-3）：时机 + "回落到 persistence 水平"判据；
6. **工程缺口补齐**（B P1-3/P1-4）：latest.pth 快照化验证、验证 GPU 2/3、环境变量、冒烟目录隔离、冒烟样本数 1600（100 步）；
7. **git/退出检测/S_churn 表述修正**（B P2-1/P2-3/P2-4）：本地已与 origin 同步；三重退出确认；S_churn 复现需显式传参；
8. **init-from 健壮性**（B P2-5）：barrier + 报错包装；锚点过渡像素实测为 0，仅记录不改代码。
9. **v2.1：best_model.pth 记录机制**（§1.6 离线 epoch 监视器）：训练进程零改动、验证走 GPU 2/3、overall RMSE 选优、原子复制；顺带自动采集验收门数据。
