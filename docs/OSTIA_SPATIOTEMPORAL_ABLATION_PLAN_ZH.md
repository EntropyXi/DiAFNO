# OSTIA 时空条件与 IAFNO 小规模架构消融计划

## 1. 目标与边界

本任务在 `OSTIA_SST` 当前已验证的数据、训练、验证和 checkpoint 语义基础上，新建独立代码分支并完成以下工作：

1. 为 7 日 SST 条件增加不泄漏未来信息的位置与季节特征；
2. 对 `patch_size: 8 -> 4`、`num_blocks: 8 -> 2/1`、`implicit_layer: 2 -> 4` 进行可归因的小规模架构消融；
3. 保持 7 日输入、15 日输出、逐日滑窗、时间顺序 70%/20%/10% 划分、mask-aware loss、lead-standardized residual/centered innovation 语义不变；
4. 本地完成实现和测试，推送到远程后在服务器独立工作树中进行数据审计、显存探测、烟测和短程消融。

本任务不删除、不覆盖现有 checkpoint、训练日志或实验目录，不在架构选择阶段使用 test split，也不把新输入架构错误地从旧 8 通道 checkpoint 续训。

## 2. Git 隔离策略

- 新分支：`codex/ostia-spatiotemporal-ablation`
- 基线：执行前 fetch 后的 `origin/OSTIA_SST`，记录精确 commit SHA；当前已知基线为 `046e34f`。
- 当前工作区存在用户修改和未跟踪文件，不在原工作区直接切分支。
- 本地使用独立 worktree：`D:\DataHub\深度学习代码\DiAFNO-spatiotemporal-ablation`。
- 服务器使用独立 worktree：`/data2/user/zzx/exam_preprocessed/DiAFNO_spatiotemporal_ablation`。
- 任何脚本都不得复用或清空现有 `experiments/ostia_7day_to15day*` 目录。

建议提交拆分：

1. `feat: add geospatial and seasonal OSTIA conditioning`
2. `test: verify spatiotemporal conditioning and checkpoint contracts`
3. `experiment: add controlled OSTIA architecture ablation runner`

## 3. 条件特征数据契约

### 3.1 模式与通道

保留旧模式 `sst_mask`，新增模式 `sst_mask_geo_season`。新模式的通道顺序必须固定并写入 checkpoint/sidecar：

| 通道 | 名称 | 定义 |
|---:|---|---|
| 0--6 | `sst_tminus6` ... `sst_t0` | 前 7 日标准化 SST |
| 7 | `valid_mask_t0` | 最后一个输入日的有效海洋 mask |
| 8 | `sin_lat` | 纬度弧度的正弦 |
| 9 | `cos_lat` | 纬度弧度的余弦 |
| 10 | `sin_lon` | 经度弧度的正弦 |
| 11 | `cos_lon` | 经度弧度的余弦 |
| 12 | `sin_doy` | 初始化日年周期正弦 |
| 13 | `cos_doy` | 初始化日年周期余弦 |

因此旧模式 `cond_chans=8`，新模式 `cond_chans=14`。模型构建时必须根据 condition schema 校验通道数，不能只相信手工填写的 `cond_chans`。

### 3.2 位置编码

- 经纬度来自 HDF5 的 `lat`、`lon` 数据集；不得从 `spatial_index` 猜测。
- 服务器数据审计后，只实现经过验证的真实 shape，同时为合成测试保留明确支持的 shape。
- 经度使用 sin/cos，避免 `-180/180` 经线断点。
- 纬度也使用 sin/cos，统一为有界球面编码。
- 非有限经纬度必须明确报错或按经过测试的 mask 规则处理，禁止静默传播 NaN。
- 静态位置特征应缓存，避免每个时间窗口重复读取和计算。

### 3.3 季节编码

- 季节相位只使用最后一个输入日 `t0`，不能读取未来 15 日 SST 或目标 mask。
- `sin_doy/cos_doy` 广播到整个空间 patch。
- 按真实 Gregorian 年长度处理 365/366 日。
- 时间解码必须 fail-closed：优先读取 HDF5 的 `units`、`calendar`、`first_date` 等可证明元数据；缺少日期语义时直接报错，禁止擅自把整数解释为 Unix timestamp。
- 需要在 checkpoint/sidecar 中记录时间解码方式和 schema version。

### 3.4 归一化与缺失值

- SST 继续仅使用 train split 统计量标准化。
- 无效 SST 继续填训练均值，因此模型空间值为 0。
- mask 保持 0/1。
- 经纬度与季节 sin/cos 已位于 `[-1,1]`，不混入 SST mean/std，也不从 val/test 估计统计量。
- train、validation、inference 必须调用同一条件构造逻辑，禁止复制三套实现。

## 4. 配置与 checkpoint 兼容性

建议新增并持久化以下语义字段：

- `condition_mode`
- `condition_schema_version`
- `condition_channel_names`
- `cond_chans`
- `calendar_encoding`
- 经纬度单位和 shape 摘要

兼容规则：

1. 旧 checkpoint 缺少新字段时，按 `sst_mask/v1/8 channels` 解释；
2. 旧 checkpoint 仍可用旧模式验证和推理；
3. 8 通道 checkpoint 不能 resume 或 init 到 14 通道模型；
4. `patch_size`、`num_blocks`、`implicit_layer` 不同的 checkpoint 必须拒绝 resume；
5. 新实验必须使用新输出目录并从头训练；
6. 错误必须在模型或优化器状态加载前给出清晰原因。

## 5. 需要修改的模块

主要修改范围：

- `diafno/data/ostia.py`：HDF5 形状验证、日期解码、位置缓存、季节相位和统一 condition builder；
- `diafno/models/config.py`：condition schema、自动/严格通道校验以及 checkpoint round-trip；
- `diafno/training/config.py`：新增 condition mode 与显式架构配置入口；
- `diafno/training/artifacts.py` 和语义 sidecar：新旧 checkpoint 兼容与拒绝规则；
- `diafno/evaluation/*`、`diafno/inference/*`：从 checkpoint 恢复同一数据契约；
- `configs/`：新增控制组与 A1--A5 配置；
- `scripts/`：新增非破坏性的显存探测、烟测和短程消融 runner；
- `tests/`：新增数据、配置、checkpoint、训练/验证一致性测试；
- `README.md` 或独立 runbook：记录使用方式与不可 resume 的原因。

不要改变现有默认 `sst_mask` 的逐值行为。新增模式必须通过显式配置启用。

## 6. 自动化测试

### 6.1 数据测试

使用小型合成 HDF5 覆盖：

- 新 condition shape 为 `[14,H,W,1]`；
- 通道顺序和数值定义准确；
- 179 度与 -179 度附近经度编码连续；
- 南北纬编码正确；
- 平年、闰年和跨年窗口正确；
- 季节特征取最后输入日，而非 target 日；
- 无效 SST 填均值后为零；
- target/target_mask 与新静态特征无耦合；
- `__getitem__` 和 `__getitems__` fast path 逐值一致；
- train/val/test 时间边界不变；
- 旧 `sst_mask` 模式与修改前逐值一致。

### 6.2 模型与 checkpoint 测试

- deterministic、diffusion、centered diffusion 均可接受 14 通道输入；
- forward/backward 在 CPU 小尺寸张量上 finite；
- checkpoint 保存、加载后 schema 完整；
- 8/14 通道错配明确失败；
- 架构参数错配明确失败；
- 训练和验证对同一样本构造的 condition 完全相同；
- resume 后 epoch/global_step 连续；
- 现有测试全部继续通过。

## 7. 服务器数据预检

在启动任何训练前，对真实 HDF5 只读输出并保存 manifest：

- 文件大小、路径和可用时的校验摘要；
- `sst/mask/lat/lon/time` 的 shape、dtype、chunks、compression；
- 文件和数据集 attributes；
- time 首尾、相邻差分及可解码日期；
- `samples_per_day`、空间索引数量；
- lat/lon min/max、非有限值数量；
- 多个 spatial index 和多天样本的坐标一致性。

只有日期语义、经纬度单位和 shape 得到证明后才能进入烟测。

A0--A5 必须绑定同一份日期 manifest。A0 虽然不把日期作为模型输入，
仍须用 manifest 排除跨缺口窗口；否则固定 seed 会在 A0 与 A1 中映射到
不同验证样本，RMSE 不具备配对可比性。manifest 身份随 checkpoint 保存并在
resume、validation、inference 时复核。

## 8. 架构消融矩阵

采用逐项隔离而不是一次性改变全部参数：

| ID | Geo/season | Patch | Blocks | Implicit | 目的 |
|---|---|---:|---:|---:|---|
| A0 | 否 | 8 | 8 | 2 | 同代码路径控制组 |
| A1 | 是 | 8 | 8 | 2 | 单独测位置/季节收益 |
| A2 | 是 | 4 | 8 | 2 | 单独测 patch 8 -> 4 |
| A3 | 是 | 4 | 2 | 2 | 测 blocks 8 -> 2 |
| A4 | 是 | 4 | 1 | 2 | 测 blocks 8 -> 1 |
| A5 | 是 | 4 | A3/A4 胜者 | 4 | 测 implicit 2 -> 4 |

若结果显示明显交互，再补 A6：`patch=8, best_blocks, implicit=4, geo/season=on`。A6 不是第一轮必跑项。

所有配置固定：

- seed；
- 数据 split；
- 同一份真实日期 manifest 与跨缺口窗口过滤；
- 样本索引/采样计划；
- optimizer、scheduler、有效 batch；
- 验证样本及顺序；
- target semantics；
- persistence 计算方式。

## 9. 显存与吞吐预检

`patch 8 -> 4` 会把二维 token 数从 `56x56=3136` 增加到 `112x112=12544`，约为 4 倍。不得直接假定旧 micro-batch 可用。

每个架构依次探测 batch 1、2、4、8；每次至少执行 warm-up 和多个正式 forward/backward，记录：

- allocated/reserved/peak GPU memory；
- 秒/iteration 与秒/optimizer step；
- 是否出现 OOM、NaN、梯度异常；
- 数据等待时间和 GPU utilization。

使用梯度累积保持 global effective batch=32。例如双卡：

- batch/GPU 8，accumulation 2；
- batch/GPU 4，accumulation 4；
- batch/GPU 2，accumulation 8。

OOM 时只降低 micro-batch，不改变有效 batch、学习率或优化步数定义。

## 10. 分阶段运行

### Stage 1：功能烟测

每个配置运行 50 optimizer steps，保存 checkpoint 后 resume 10 steps，再对固定 16 个 validation 样本推理。

硬门槛：

- loss、gradient、prediction 全部 finite；
- checkpoint 可加载；
- resume 从正确 global_step 开始；
- condition schema 与形状一致；
- 无异常 I/O 停顿或显存持续增长。

### Stage 2：快速筛选

每个存活配置运行 300 optimizer steps并评估固定 val-200，输出：

- overall RMSE、MAE、bias、correlation；
- Day +1、+7、+15 RMSE；
- persistence RMSE；
- skill vs persistence；
- 峰值显存；
- 秒/optimizer step 和预计 epoch 时间。

300 步只用于淘汰明显较差或不可运行的配置，不能据此宣布最终最优。

### Stage 3：稳定排名

保留 2--3 个最佳配置，继续到 1500 optimizer steps，在 500/1000/1500 步使用相同 val-200 复评。若排名不稳定，将前两名延长到 5000 optimizer steps。

对最终候选使用 paired temporal block bootstrap 计算相对 persistence skill 的 95% CI。

### 选择规则

1. 首要指标：validation overall RMSE 和 skill vs persistence；
2. 第二指标：Day +15 RMSE；
3. 差异小于 0.5% 时优先速度快、显存低的方案；
4. bootstrap CI 不支持明确差异时视为统计持平；
5. test split 在架构冻结前禁止使用。

## 11. 从 deterministic 到 centered diffusion

架构初筛优先使用 deterministic residual IAFNO，原因是速度快、RMSE 信号稳定且不需要对所有候选支付扩散采样成本。

获胜后：

1. 用获胜 schema/架构训练新的 deterministic mean；
2. 仅使用 train split 重算 deterministic lead residual statistics；
3. 用新 deterministic mean 生成 centered innovation statistics；
4. 从头训练同 schema/架构的 centered diffusion；
5. 先完成 centered smoke 和 resume continuity；
6. 再做小批量 CRPS、spread-skill、coverage、8/16 成员收敛评估。

旧 frozen mean 和旧 centered diffusion checkpoint 均不得复用，因为输入通道和/或 backbone shape 已改变。

扩散晋级条件：

- ensemble mean RMSE 不明显劣于新 deterministic mean；
- CRPS 相对 deterministic/persistence 有改善；
- spread-skill 与 coverage 没有严重失配；
- 结果不是由极小样本偶然造成。

## 12. 实验目录与可复现性

服务器输出根目录：

```text
experiments/ostia_spatiotemporal_ablation/
  A0_baseline_p8_b8_i2/
  A1_geo_p8_b8_i2/
  A2_geo_p4_b8_i2/
  A3_geo_p4_b2_i2/
  A4_geo_p4_b1_i2/
  A5_geo_p4_best_i4/
  manifests/
  summary/
```

每个 run 保存：

- 完整展开配置；
- branch 和 commit SHA；
- HDF5 路径与只读结构摘要；
- condition schema；
- seed 与样本计划；
- stdout/stderr 日志；
- checkpoint 和语义 sidecar；
- validation JSON；
- 显存与吞吐结果。

runner 遇到非空输出目录必须拒绝启动，不自动删除、不覆盖、不复用旧结果。

## 13. 验收标准

代码完成的最低标准：

- 本地全部相关测试通过；
- 旧模式回归测试通过；
- 真实 HDF5 schema 已验证；
- A0/A1 至少完成 smoke，证明新增通道端到端可训练和验证；
- patch=4 至少完成一次经过显存探测的 forward/backward；
- 每个配置有不可变 manifest；
- 无 checkpoint、数据文件、日志或临时产物进入 Git；
- diff 中没有无关重构或用户现有未提交文件。

实验完成后生成固定比较表：

| 配置 | Overall RMSE | Skill vs persistence | Day15 RMSE | Peak VRAM | sec/step | 判断 |
|---|---:|---:|---:|---:|---:|---|

## 14. 执行纪律

- 先检查、再修改、再测试、最后推送；
- 不删除任何已有文件或实验结果；
- 不在服务器直接编辑源代码；服务器只 fetch/checkout 并产生运行产物；
- 不为通过测试而放宽 checkpoint 或数据语义检查；
- 不把短烟测 loss 当作泛化结论；
- 不用 test 集挑选配置；
- 遇到数据语义无法证明、旧 checkpoint 混用、数值异常或持续 OOM 时停止对应阶段并保留诊断证据。
