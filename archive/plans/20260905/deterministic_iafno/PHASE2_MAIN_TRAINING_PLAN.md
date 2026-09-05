# Phase 2：Frozen-Mean Centered Diffusion 主训练实施计划

状态：`DRAFT_FOR_DOUBLE_HIGH_PRECISION_REVIEW`  
编写日期：2026-09-01  
适用分支：`OSTIA_SST`  
计划基线提交：`4f293af`  
目标：在 12 小时窗口内完成设计审查、实现、验证、旧产物归档，并让正确的新主训练在 tmux 中稳定运行。

---

## 0. 执行者须知

本文件面向不具备此前对话上下文的实现 agent。执行前必须完整阅读本文件、
`deterministic_iafno/RUNBOOK.md`、
`deterministic_iafno/reports/PHASE0_PHASE1_FINAL_REPORT_ZH.md`，不得只凭文件名猜测任务。

硬约束：

1. 任务始终是**连续 7 日 SST 预测随后 15 日 SST**，不是月尺度任务。
2. Phase 1 已选定且冻结的条件均值为：
   `experiments/det_lead_standardized/epoch_015.pth`。
3. 冻结均值 checkpoint SHA-256 必须是：
   `cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6`。
4. 新扩散模型只学习 centered innovation，不重新联合训练均值模型。
5. centered statistics 只能由 train split 计算；val/test 不得进入训练统计。
6. val 只用于选择与监测；已经冻结的 test-1000 不再用于调参或选择。
7. 所有源码只在本地工作区修改；通过 Git push 后，服务器只允许 `git pull --ff-only` 获取源码。
8. 命令行统一使用 Git Bash；只有 Git Bash 无法完成的 Windows UI 操作才允许使用其他 shell。
9. 不删除旧 checkpoint、配置、日志或评估结果。旧主训练目录必须先完成可恢复归档，才能复用规范输出位置。
10. 不终止、迁移或抢占其他用户 GPU 进程。GPU 选择必须基于实时观测。
11. 任何 checkpoint 语义、目标空间、anchor/mean 加法次数不明确时必须 fail closed。
12. 审查通过前不得把实现提交、推送或部署到服务器。

---

## 1. 已确认的权威现状

### 1.1 数据与任务

- HDF5：`/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5`
- `sst=(1126100,1,448,448)`
- `mask=(1126100,448,448)`
- 每个日期 100 个空间 patch。
- 一个样本覆盖 22 日：7 日条件 + 15 日预测。
- 时间划分：按日期顺序 70% train、20% val、10% test。
- `condition_mode=sst_mask`，即 7 个 SST 通道 + 最后输入日 mask，共 8 个条件通道。

### 1.2 Phase 1 冻结均值

选中模型：`det_std_epoch015`。

- 模型：deterministic IAFNO
- `target_mode=residual`
- `target_scaling=lead_standardized`
- 输入 7 日，输出 15 日
- val-200 RMSE：1.1370339337 K
- val persistence RMSE：1.1870674505 K
- val MSE skill：8.2521%
- test-1000 RMSE：0.6306998216 K
- test persistence RMSE：0.7169742445 K
- test MSE skill：22.6183%
- 22 日 paired block bootstrap 下，val/test 的 15 个 lead RMSE 差 95% CI 均低于 0。

此 checkpoint 是新主训练的冻结条件均值，不再进行 optimizer update。

### 1.3 旧主训练位置

旧规范目录：

`experiments/ostia_7day_to15day_residual_scratch`

服务器盘点时约 2.4 GB，包含旧 diffusion checkpoint、训练曲线、验证/测试 JSON 等。
旧 `ostia_train` tmux 已完成 35 epoch，目前停留在 shell prompt，不再写 checkpoint。

新 centered 主训练在归档完成后继续使用这个规范目录，以便替代旧主训练的位置；
旧目录本身必须整体迁入 archive，绝不覆盖。

### 1.4 服务器资源现状（仅为计划编写时快照）

- `/data2`：约 7.0 TB，总用量约 2.0 TB，可用约 4.7 TB。
- 编写计划时 0–7 卡均有其他用户计算：0 为 gwp，1–4 为 lz，5–7 为 hjc。
- 该快照不能作为后续启动依据；启动前必须重新连续采样 GPU 状态。

---

## 2. 主训练的数学定义

### 2.1 符号

数据集已有全局 train normalization：

`x = (SST - sst_mean) / sst_std`

对每个样本：

- 条件：`c = [x_(t-6), ..., x_t, mask_t]`
- anchor：`a = x_t`
- 第 `l` 个未来日真实残差：`r_l = x_(t+l) - a`，`l=1..15`
- 冻结确定性均值：`mu_l = F_mean(c)`
- centered innovation：`e_l = r_l - mu_l`

只在 train split 上估计每个 lead 的 innovation 均值与标准差：

- `m_l = E_train[e_l]`
- `s_l = Std_train[e_l]`
- `z_l = (e_l - m_l) / s_l`

扩散模型学习：

`p(z | c)`

采样与还原：

1. `z_hat = Diffusion.sample(c)`
2. `e_hat_l = z_hat_l * s_l + m_l`
3. `r_hat_l = mu_l + e_hat_l`
4. `x_hat_(t+l) = a + r_hat_l`
5. `SST_hat = x_hat * sst_std + sst_mean`

### 2.2 必须锁死的代数不变量

1. anchor 只在最终绝对 SST 重建时加一次。
2. deterministic mean 只在 centered innovation 还原时加一次。
3. 训练 target 必须是 `target - anchor - frozen_mean`，之后才做 innovation lead standardization。
4. frozen mean 的 `predict()` 输出已经从其自身的 lead-standardized 空间反变换到 normalized residual 空间；不得再次应用旧 lead stats。
5. centered diffusion 的 `lead_mean/lead_std` 指 innovation stats，不是 Phase 1 的 raw residual stats。
6. innovation 标准化后使用 `sigma_data=1.0`；不得沿用旧 residual diffusion 的 `sigma_data=0.15`。
7. mask 必须应用在未来 15 日对应的同一有效像素上。

---

## 3. 目标软件结构

实现应保持旧 diffusion 与 deterministic 路径完全兼容，并新增明确的第三种模型类型：

`model_type=centered_diffusion`

### 3.1 新增文件

建议新增：

1. `deterministic_iafno/centered_diffusion.py`
   - `FrozenMeanCenteredDiffusion`
   - 包含一个冻结 deterministic mean 和一个可训练 EDM diffusion。
2. `deterministic_iafno/compute_centered_stats.py`
   - 只读 HDF5 train split。
   - 加载冻结 mean checkpoint。
   - 计算每个 lead 的 centered innovation mean/std。
3. `deterministic_iafno/tests/test_centered_diffusion.py`
4. `deterministic_iafno/tests/test_centered_stats.py`
5. `deterministic_iafno/tests/test_centered_checkpoint_roundtrip.py`
6. `configs/ostia_centered_diffusion_main.json`
   - 作为人可读、机器可读的主训练配置源。
7. `scripts/archive_legacy_ostia_main.sh`
   - 默认 dry-run，只有显式 `--execute` 才进行归档。
8. `scripts/run_ostia_centered_main.sh`
   - 只读取已冻结配置，执行 preflight 后启动。
9. `scripts/smoke_ostia_centered.sh`
10. `scripts/watch_ostia_centered.sh`

### 3.2 `FrozenMeanCenteredDiffusion` 的职责

该 wrapper 的每一条职责都是可被单元测试锁死的契约：

- 注册 `mean_model` 与 `diffusion` 两个子模块。
- 构造后立即 `mean_model.requires_grad_(False)`；wrapper 覆写 `train(mode=True)`：先
  `super().train(mode)`，随后强制 `self.mean_model.eval()` 并返回 `self`（DDP 包装后同样生效）。
- `forward(residual_target, condition, target_mask)`：
  - 在 fp32、no-grad 下调用 `mean_model.predict(condition)` 得到 frozen mean `mu`；其输出已是
    normalized residual 空间，禁止再应用任何 lead stats，禁止改用 `_network_prediction`；
  - 在 fp32 下计算 `e = residual_target - mu`、`z = (e - m) / s`；此段必须置于 autocast
    之外或显式 fp32，防止 fp16 相消误差；
  - 仅将 `z` 传入 EDM diffusion loss（diffusion 部分可用 AMP）。
- `sample(condition, ...)`：采样 `z_hat` → `e_hat = z_hat * s + m` →
  `r_hat = mu + e_hat`；返回 normalized residual forecast `r_hat`，不加 anchor、不反标准化回 SST。
- sampler 属性必须读写双向委托给内部 diffusion：`S_churn`（赋值必须改变
  `diffusion.S_churn`，读取必须反映内部值，否则 evaluator 的 `--s-churn` 会静默失效）、
  `sigma_min`、`sigma_max`、`rho`、`num_sample_steps`；wrapper 的
  `sample(condition, num_sample_steps=None, seed=None)` 把参数转发给内部 `diffusion.sample`。
- 不暴露 `preconditioned_network_forward`；probe 模式对 centered 无意义，evaluator 会按
  `model_type != "diffusion"` 拒绝。
- mean 权重必须以 `mean_model.*` 等前缀进入 centered checkpoint 的 state dict，使推理/恢复不依赖
  原 mean 文件路径；mean 的 lead stats 是 `persistent=False` buffer、不在 state dict 内，必须由语义层另行携带。

### 3.3 配置字段

`OSTIAModelConfig` 新增或明确（新字段必须同步进入 `to_checkpoint()`、`from_checkpoint()` 与
`MODEL_IMMUTABLE_FIELDS`，缺一视为实现未完成）：

- `model_type=centered_diffusion`
- `target_mode=residual`
- `target_scaling=lead_standardized`（centered 语义）
- `lead_mean/lead_std`：centered innovation stats（15 项）
- `mean_lead_mean/mean_lead_std`：冻结 deterministic mean 自身的 residual lead stats（15 项）
- `mean_checkpoint_sha256`：冻结 mean 身份

训练配置新增 `mean_checkpoint_path`（仅 fresh run 用）、`centered_stats_path`（fresh run 必须提供）；
CLI 新增 `--mean-checkpoint`、`--centered-stats`；`--model-type` choices 增加
`centered_diffusion`；`validate_lead_stats_dict` 新增
`target_space=normalized_centered_residual` 校验分支；centered 模式下显式拒绝 `--init-from`，
`--lead-stats` 与旧校验互斥。

规则（全部 fail closed）：

- fresh centered run 缺 `mean_checkpoint_path` 或 `centered_stats_path` 任一文件必须失败。
- fresh run 必须在每个 rank 独立校验，任一失败全体退出：
  1. mean checkpoint 文件 SHA-256 等于 stats JSON 的 `mean_checkpoint_sha256`，且等于锁定值
     `cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6`；
  2. mean checkpoint sidecar manifest 存在，其 immutable 中 `model_type=deterministic`、
     `target_mode=residual`、`target_scaling=lead_standardized`、`input_days=7`、
     `output_days=15`，且 `lead_mean/lead_std` 与 stats JSON 的
     `mean_lead_mean/mean_lead_std` 完全一致；
  3. mean checkpoint 架构字段 `input_days/output_days/cond_chans/target_chans/image_size/patch_size/`
     `embed_dim/num_blocks/explicit_layer/implicit_layer/hidden_size_factor` 与 centered 配置完全一致；
     mean 自身无意义的 `sigma_data/P_mean/P_std` 不在比较范围；
  4. stats JSON 的 `split=train`、`target_space=normalized_centered_residual`、15 个 lead、
     `lead_std` 全部有限且大于 0。
- fresh run 以 centered stats JSON 为 mean stats 的唯一事实来源构造 mean 模型；mean sidecar 仅交叉校验。
- resume centered run 从自身 checkpoint/sidecar 恢复语义，不要求原 mean 路径仍存在。
- `mean_checkpoint_sha256` 与 `mean_semantics_sha256` 必须与 model config/centered stats 一致。

### 3.4 checkpoint 与 resume

建议把新 checkpoint schema 提升到 4，但保留 schema 3 的只读/恢复兼容。

immutable 语义至少包含：

- 原有架构/数据/目标字段
- `model_type`
- centered innovation `lead_mean/lead_std`
- `mean_lead_mean/mean_lead_std`
- `mean_checkpoint_sha256`
- `mean_semantics_sha256`
- `split=train`
- `condition_mode=sst_mask`

resume 必须：

- 先从 sidecar 恢复 immutable 语义（含 `model_type`、innovation `lead_mean/lead_std`、
  `mean_lead_mean/mean_lead_std`、`mean_checkpoint_sha256`、`split=train`、
  `condition_mode=sst_mask`）再 build model；`restore_resume_semantics` 的字段定位逻辑必须能定位新增字段。
- 严格恢复 frozen mean、diffusion、optimizer、scheduler、scaler、随机状态。
- mean hash 或 centered stats 不一致时拒绝恢复（每个 rank 独立校验）。
- 仍允许现有的显式、经审查 compatible override，但不得覆盖 immutable centered 语义。

推理路径同样自包含：`InferenceModelLoader.load` → `OSTIAModelConfig.from_checkpoint` →
`build_model` 必须仅凭 centered checkpoint 内的 config + state dict 重建 wrapper；innovation stats、
mean residual stats、mean 权重全部来自 checkpoint 本体，evaluator 不读取任何外部 mean/stats 文件。
schema 提升到 4，schema 3 保持只读/恢复兼容；旧 checkpoint 无新字段时不校验新字段。

### 3.5 optimizer 与 AMP

- optimizer 只接收 `requires_grad=True` 的 diffusion 参数（先冻结 mean 再构造 AdamW，或按
  `requires_grad` 过滤）。
- frozen mean 的 `.grad` 必须始终为 `None`（首个 optimizer step 后由断言确认）。
- 保留 AMP + GradScaler。
- 新增 `skipped_optimizer_steps`：比较 `scaler.step` 前后的 scale/inf 状态，记录 overflow skip 及发生 step；
  checkpoint/history 保存 skip count。
- overflow skip 时不得调用 `scheduler.step()`；LR 只随真实 optimizer step 推进；`global_step` 仍随数据处理推进并用于日志。
- 20-step 与 500–1000-step 烟测必须 `skipped_optimizer_steps == 0`；主训练出现 skip 必须记录 skip rate，
  大于 1% 需要人工评审。

---

## 4. Train-only centered statistics

### 4.1 计算协议

- split：train
- checkpoint：冻结 `det_std_epoch015`
- 样本数：优先 8192；若在时间预算内明显过慢，允许降到 4096，但必须记录决定与耗时。
- batch：32；使用 chunk-aware indices，将同一 initialization date 的连续空间 patch 聚合读取。
- AMP：允许推理 AMP，但 accumulator 必须 float64。
- 每 lead 独立累计 masked sum、squared sum、count。

### 4.2 输出 JSON 必须包含

- `split=train`
- `target_space=normalized_centered_residual`
- `input_days=7`
- `output_days=15`
- `condition_mode=sst_mask`
- `num_samples`、`dataset_size`
- `selection` 与确定性 index 生成说明（复用 `compute_lead_stats.py` 的 chunk-aware 逻辑，无随机 seed）
- `indices_sha256`
- `mean_checkpoint`
- `mean_checkpoint_sha256`
- `mean_semantics_sha256`：mean sidecar `semantic_manifest.immutable` 规范化 JSON 的 SHA-256
- `mean_lead_mean/mean_lead_std`：必须从 mean checkpoint sidecar 读取并复制进本 JSON，与 sidecar
  断言一致；mean checkpoint 缺 sidecar 时 fail closed
- `sst_mean/sst_std`
- 15 个 innovation `lead_mean`
- 15 个正且有限的 innovation `lead_std`
- overall innovation std（诊断用途）
- 每 lead 有效像素数

### 4.3 stats 验收

- 两次相同命令输出逐位一致（同机器同环境；跨环境相对容差小于 `1e-12`）。
- 不包含任何 val/test 元数据或索引。
- 15 个 std 全部有限且 >0；标准化后 pooled mean≈0、per-lead std≈1。该数值项是近似恒等式，
  只能验证实现一致性，不能替代 split 溯源；防泄漏硬证据是 `split=train`、`indices_sha256`
  与 validator 的 provenance 强制校验。
- round-trip `inverse(transform(e)) == e` 最大误差 <1e-6；另做真实重建诊断：train 样本上分别用
  `z=0` 与实际 `z` 计算 `anchor + mu + m + s*z` 并对比绝对 target，误差在 fp32 容差内。
- `mean_checkpoint_sha256` 与锁定值一致；`mean_semantics_sha256` 与 mean sidecar 一致；
  `mean_lead_mean/mean_lead_std` 与 mean sidecar 完全一致。
- 新 validator（由 `test_centered_stats.py` 锁定）必须拒绝：`split != train`、target_space 缺失/错误、
  lead 数量错误、std 非有限/非正、mean SHA 与锁定值不符、mean stats 与 mean sidecar 不一致、
  JSON 含 val/test 键。

---

## 5. 旧主训练归档与规范位置替换

### 5.1 归档目标

归档根目录：

`experiments/archive/pre_centered_20260901`

旧目录移动到：

`experiments/archive/pre_centered_20260901/legacy_ostia_7day_to15day_residual_scratch`

新主训练重新创建：

`experiments/ostia_7day_to15day_residual_scratch`

### 5.2 归档前置检查

脚本必须 fail closed：

1. 当前 repo HEAD 是审查通过后的指定提交。
2. 旧目录存在且不是 symlink。
3. archive 目标不存在；若存在直接退出，不做覆盖或合并。
4. `realpath` 确认源和目标都在 repo 的 `experiments` 下。
5. `pgrep -af` 和 tmux pane 均确认没有进程写旧目录。
6. `/data2` 可用空间足够；同盘 `mv` 不复制 2.4 GB 数据。
7. 保存旧 tmux pane 最后 500 行。
8. `experiments/archive` 根与目标路径均不在任何 symlink 链中；realpath 全部落在 repo 的
   `experiments` 下。
9. 新规范目录在归档完成前不存在；若存在且非本次重建，中止并人工处理。

### 5.3 归档动作

执行顺序：

1. 创建 archive 根目录。
2. 生成旧目录文件清单、大小、mtime。
3. 对所有配置、sidecar、JSON、NPZ、PNG、日志和 `.pth` 生成 SHA-256。
4. 保存：
   - `source_git_commit.txt`
   - `legacy_tmux_tail.txt`
   - `legacy_file_manifest.tsv`
   - `legacy_sha256.txt`
   - `ARCHIVE_README.md`
5. 使用同文件系统原子 `mv`，不使用 `rm`、覆盖移动或通配目标。
6. 移动后逐项验证文件数、总大小和 SHA-256。
7. 只有验证通过后才创建新的规范输出目录。

### 5.4 新规范目录初始化

新目录必须先包含：

- `config/ostia_centered_diffusion_main.json`
- `config/ostia_centered_diffusion_main.json.sha256`
- `config/centered_stats_train.json`
- `config/centered_stats_train.json.sha256`
- `config/frozen_mean.pth`
- `config/frozen_mean.pth.semantics.json`
- `config/frozen_mean.sha256`
- `config/source_git_commit.txt`
- `config/launch_command.sh`
- `logs/`

训练代码将 checkpoint 写在输出目录根，不额外嵌套 `checkpoints/`。

冻结 mean 可复制到新目录以保证运行自包含；复制后 SHA-256 必须匹配锁定值。

### 5.5 回滚

在新主训练尚未写 checkpoint 前，如归档或初始化失败：

- 不删除新目录；先重命名为带 `_failed_<timestamp>` 的诊断目录。
- 将 archive 中旧目录原子移动回原规范位置。
- 校验 SHA-256。

一旦新训练已产生 checkpoint，不自动回滚；保留新旧两套目录并人工决定。

---

## 6. 测试矩阵

### 6.1 本地单元测试

新增测试至少覆盖：

1. toy tensor 的 centered target 代数正确。
2. anchor 与 mean 都只加一次。
3. `mean_model` 参数不产生 grad，diffusion 参数产生 grad。
4. wrapper `.train()` 后 mean 仍为 eval。
5. innovation transform/inverse round-trip。
6. sample 返回 normalized residual，不返回绝对 SST。
7. zero innovation 精确退化为 deterministic mean。
8. checkpoint state dict 包含 frozen mean，并可在不存在原 mean 路径时 round-trip。
9. fresh run mean SHA 不匹配时 fail closed。
10. centered stats target_space/split/lead 数量/有限性验证。
11. centered checkpoint bare resume 恢复全部语义。
12. 显式冲突 resume 失败。
13. legacy diffusion 与 deterministic golden tests 数值不变。
14. evaluator 对 centered model 只 re-anchor 一次。
15. CPU tiny backbone forward/backward/sample。
16. wrapper 的 `S_churn/sigma_min/sigma_max/rho/num_sample_steps` 读写双向委托有效。
17. AMP overflow skip 时 optimizer 与 scheduler 都不推进，skip count/global-step 日志语义正确。

验收：完整 `deterministic_iafno/tests` 全绿；不是只跑新测试。

### 6.2 服务器 CPU/导入检查

- `python -m compileall diafno deterministic_iafno`
- 完整 unittest
- config JSON schema/字段校验
- checkpoint hash 校验
- HDF5 只读 dataset 取 2 个样本并验证 shape/metadata

### 6.3 GPU 结构烟测

GPU 优先级：1–4 中实时显存占用最低且连续低利用率的一张。

烟测使用独立输出目录 `experiments/ostia_centered_smoke_scratch`，任何阶段不得写入规范目录或
archive；烟测目录最终保留，不删除。

第一阶段 smoke：

- 单卡
- batch-per-GPU=2 或 4
- effective batch 保持 8 或 16 即可，目的不是可比训练
- 20 optimizer steps
- no test split

必须验证：

- 前 20 step loss 全部有限
- prediction/target/mask shape 正确
- mean grad 全为 None
- diffusion 至少一个参数发生更新
- peak memory 有余量
- 保存 latest checkpoint + sidecar
- 从 latest resume 再跑至少 2 step，global step 连续
- 4-step sampler 对 2 个 val 样本生成有限输出
- `skipped_optimizer_steps == 0`
- `--mean-checkpoint` / `--centered-stats` 每 rank 校验通过

第二阶段正式烟测：

- 500–1000 optimizer steps；优先完整 1000 step，只有逼近 12 小时启动期限时才允许在不少于 500 step 后进入主训练
- 使用与正式训练相同 target、stats、sigma_data、LR、effective global batch（32）
- 全部 loss 有限、`skipped_optimizer_steps == 0`、resume 连续、loss EMA 下降或不系统性爆炸、
  mean grad 为 None、diffusion 参数持续更新
- 不把 pilot loss 与 deterministic MSE 直接比较

主训练前必须同时满足：20-step 结构检查、checkpoint resume、有限 sample、
不少于 500 step 的正式烟测均通过。默认完成 1000 step；若只完成 500–999 step，
必须在 launch manifest 中记录缩短原因与实际步数。

---

## 7. 正式主训练配置

冻结配置建议如下，审查后才可最终锁定：

| 字段 | 值 |
|---|---|
| model_type | `centered_diffusion` |
| target_mode | `residual` |
| target_scaling | `lead_standardized`（innovation stats） |
| frozen mean | `det_std_epoch015` + 锁定 SHA-256 |
| sigma_data | `1.0` |
| sigma_min | `0.002` |
| sigma_max | `80.0` |
| P_mean / P_std | `-1.2 / 1.2` |
| rho | `7.0` |
| sampling_steps | `16` |
| epochs | `35` |
| samples_per_epoch | `31200` |
| learning_rate | `2e-4` |
| min_learning_rate | `1e-6` |
| weight_decay | `1e-4` |
| max_grad_norm | `1.0` |
| AMP | 开启 |
| checkpoint_interval | `1`（前期诊断优先） |
| seed | `123` |
| num_workers | 每 rank `2`，稳定后可调至 `4` |
| prefetch_factor | `1` |

`configs/ostia_centered_diffusion_main.json` 是唯一权威配置源，包含上表全部字段（含
`sigma_data=1.0`）；launcher 把 JSON 逐字段展开为 CLI，或 trainer 新增 `--config` 直接读取。
启动前打印并校验 JSON 与 CLI 展开一致；centered 运行必须拒绝 `sigma_data != 1.0`。训练配置工厂
默认是 0.15，遗漏 `--sigma-data 1.0` 属于 launch 失败，不是可继续的警告。

### 7.1 默认双卡

由于 wrapper 同时驻留 frozen mean 与 diffusion，初始采用保守显存配置：

- 2 GPUs
- batch-per-GPU=8
- gradient-accumulation=2
- effective global batch = `2 × 8 × 2 = 32`
- optimizer steps/epoch = `31200 / 32 = 975`

启动模板（卡号由 preflight 写入，不硬编码）：

```bash
CUDA_VISIBLE_DEVICES=<gpu_a>,<gpu_b> \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  trainer_ostia.py \
  --output-dir experiments/ostia_7day_to15day_residual_scratch \
  --model-type centered_diffusion \
  --target-mode residual \
  --target-scaling lead_standardized \
  --mean-checkpoint experiments/ostia_7day_to15day_residual_scratch/config/frozen_mean.pth \
  --centered-stats experiments/ostia_7day_to15day_residual_scratch/config/centered_stats_train.json \
  --sigma-data 1.0 \
  --batch-per-gpu 8 \
  --gradient-accumulation 2 \
  --learning-rate 2e-4 \
  --num-workers 2 \
  --prefetch-factor 1 \
  --num-epochs 35 \
  --samples-per-epoch 31200 \
  --checkpoint-interval 1
```

### 7.2 单卡 fallback

当 12 小时窗口内没有两张可用卡，但有一张可用卡：

- batch-per-GPU=8
- gradient-accumulation=4
- effective global batch = 32
- 其余配置不变

不得用单卡 fallback 改变 LR 或每 epoch 样本暴露量。

### 7.3 OOM fallback

如果 wrapper 在 batch=8 OOM：

- 双卡：batch=4，gradient accumulation=4
- 单卡：batch=4，gradient accumulation=8

effective global batch 始终保持 32。每次 OOM 后先退出进程并确认显存释放；不得反复无界重启。

---

## 8. GPU 选择与 tmux

### 8.1 GPU preflight

候选优先 1–4。对每张候选卡连续采样至少 3 次、间隔 5 秒：

- `memory.used`
- `utilization.gpu`
- `power.draw`
- compute process PID、owner、command

“可用”默认定义：

- 没有其他用户 compute process；或仅有明确可忽略的显示/监控占用
- 显存 <1.5 GB
- 三次平均 GPU util <10%

不得仅凭某一瞬间 `0%` 判定空闲。

如果没有空闲卡：

- 建立只读 watcher 等待，不杀进程。
- 双卡等待到执行窗口剩余 2 小时；仍无双卡则尝试单卡。
- 若连单卡都没有，必须向用户报告资源阻塞，不得擅自与他人高利用率作业共享。

### 8.2 tmux 规范

- smoke：`ostia_centered_smoke`
- main：`ostia_centered_main`
- watcher：`ostia_centered_watch`

启动后必须立即记录：

- tmux session 存在
- shell command line
- PID / GPU mapping
- 前 10 step 日志
- 当前 git commit
- config hash
- mean checkpoint hash
- centered stats hash

主训练日志：

`experiments/ostia_7day_to15day_residual_scratch/logs/train.log`

### 8.3 “正确跑起来”的验收定义

只有同时满足以下条件，才能宣布主训练已正确启动：

1. tmux session 存活且训练 PID 存活。
2. world size 与 GPU 数正确。
3. effective global batch=32。
4. 日志明确打印 `centered_diffusion`、mean SHA、stats SHA、target algebra。
5. 前 10 optimizer steps loss 有限。
6. mean 参数无梯度，diffusion 参数有更新。
7. GPU 显存稳定，无 OOM/restart loop。
8. HDF5 step 速度进入合理范围；单个冷启动 batch 慢不判失败。
9. `latest.pth`/sidecar 能在第一个 checkpoint 时正确落盘。
10. 不存在 test evaluation 进程。
11. 前 10 步 `skipped_optimizer_steps == 0`，日志打印 config JSON hash、mean SHA、stats SHA 与 skip 计数。
12. launch manifest 记录本表 1–11 全部证据。

---

## 9. 验证与模型选择策略

主训练启动后，独立 watcher 只做 fixed val-200：

- seed=123
- s_churn=0
- sampling_steps=16
- 初始 `ensemble_members=1` 用于快速 sanity；epoch 5 后执行 16 vs 32 steps、1 vs 4 ensemble
  的廉价推理侧消融，消融完成后把选中配置写入 watcher 配置并锁定；此后所有跨 epoch 比较与
  `best_val_mean_rmse.pth` 判定统一使用该锁定配置，锁定前产生的数值不得用于跨 epoch 选择
- epoch 1/3/5 做早期诊断，之后每 5 epoch

主训练前不做大规模 EDM 参数扫描。以下参数固定：

- `sigma_data=1.0`
- `P_mean=-1.2`
- `P_std=1.2`
- `rho=7`
- `sigma_min=0.002`
- `sigma_max=80`

epoch 5 后只允许廉价推理侧消融：

- sampling steps：16 vs 32
- ensemble members：1 vs 4
- `S_churn=0` 为主；只有必要时再测一个预先记录的小正值

旧 diffusion 的 16→100 steps 仅改善约 0.45%，因此不得把大规模 sampling-step
扫描重新放回主训练前关键路径。

应报告：

- ensemble mean RMSE/MAE/bias/correlation
- persistence skill
- residual correlation/std ratio
- ensemble spread（如果 evaluator 已支持）
- centered innovation diagnostics

注意：

- centered diffusion 的单样本 RMSE 不应被期待立刻优于 deterministic mean。
- 主目标包括分布建模；需要后续增加 CRPS、spread-skill、coverage 等概率指标。
- 当前 test-1000 不参与任何 checkpoint 选择。

`best_model.pth` 的早期定义应显式区分：

- `best_val_mean_rmse.pth`
- 后续新增概率指标后再定义 `best_val_crps.pth`

不得用一个模糊的 `best_model.pth` 覆盖多个标准。

---

## 10. Git 与服务器部署流程

### 10.1 本地实现

- 所有编辑使用 `apply_patch` 或实现 agent 的安全编辑接口。
- 不碰现有未关联 dirty 文件。
- 每个逻辑阶段查看 `git diff --check`。
- 完整测试通过后才提交。

建议提交拆分：

1. `feat: add frozen-mean centered diffusion`
2. `feat: compute train-only centered innovation stats`
3. `test: cover centered diffusion semantics and resume`
4. `ops: add legacy archive and centered training launchers`
5. `docs: lock centered main training runbook`

### 10.2 Codex review gate

DeepSeek 执行 agent 完成后不得自行 push。Codex 必须审查：

- 全部 diff
- 数学目标空间
- checkpoint 自包含性
- legacy compatibility
- 测试覆盖
- 脚本路径安全
- 归档脚本 dry-run

发现问题返回同一执行对话修复，直到 review 通过。

### 10.3 Push / Pull

Codex review 通过后：

1. 本地 push 到 `OSTIA_SST`。
2. 服务器检查工作树无会被覆盖的 tracked 修改。
3. `git pull --ff-only origin OSTIA_SST`。
4. 核对服务器 HEAD 等于本地 HEAD。
5. 在服务器既有 DiAFNO 环境重新运行完整测试。

禁止在服务器直接编辑 tracked source。

---

## 11. 12 小时时间预算与决策点

从计划审查开始计时：

| 时间 | 目标 |
|---|---|
| T+0–1.5h | 双高精度计划审查、修订并获得 PASS |
| T+1.5–5h | DeepSeek-V4-Pro Max 实现；Codex 持续 review |
| T+5–6.5h | 本地完整测试、修复、提交、push |
| T+6.5–7.5h | 服务器 pull、完整测试、centered stats |
| T+7.5–9h | GPU 20-step smoke、resume、finite sample、≥500 step 正式烟测（默认 1000；不足 1000 需在 launch manifest 记录原因） |
| T+9–10h | 归档旧主训练，初始化新规范目录 |
| T+10–12h | 双卡主训练启动；无双卡则单卡 fallback |

必须保留至少 2 小时给正式启动，不得无限扩展 smoke 或非关键重构。

注：单卡 fallback 下 500 step 烟测耗时约为双卡两倍；若因此逼近窗口末端，允许在 ≥500 step
后启动并把原因与实测步数写入 launch manifest；任何情况下不得以 <500 step 的烟测换取按时启动。

如实现到 T+7h 仍未通过代数/round-trip/legacy tests，不允许为了赶时间启动错误训练；
此时优先修正 correctness，并明确报告延迟原因。

---

## 12. DeepSeek 双高精度审查要求

审查对话必须使用 `deepseek-v4-pro-max`，只审查计划，不写代码。要求执行两遍互相独立的审查：

### Pass A：正向架构与可执行性审查

- 数学目标是否正确
- 是否存在 anchor/mean/standardization 双重应用
- frozen mean 与 centered checkpoint 是否自包含
- train-only stats 是否无泄漏
- DDP/AMP/resume 是否可执行
- 12 小时排期是否现实

### Pass B：对抗性失败审查

- 假设实现者会误解计划，找出所有歧义
- 假设服务器中途断线、GPU 被占、OOM、旧目录非预期，检查回滚
- 检查是否可能覆盖旧 checkpoint
- 检查是否可能在 test 上选择模型
- 检查 smoke 通过但主训练语义仍错误的漏洞
- 检查 checkpoint 迁移、hash、sidecar 的供应链问题

审查输出固定格式：

1. `VERDICT: PASS` 或 `VERDICT: REVISE`
2. Blocking issues（编号、证据、建议修正）
3. Non-blocking improvements
4. 一份可直接替换原计划相应章节的修订文本
5. 最终 requirement-to-evidence matrix

Codex 根据审查修改本文件后，必须把修订版本再次发给同一审查对话；只有明确 `VERDICT: PASS` 才进入实现。

---

## 13. DeepSeek 执行对话要求

计划 PASS 后，在 dsh 的 DiAFNO 工作区创建**另一个新对话**，模型使用 `deepseek-v4-pro-max`。

执行 agent 的边界：

- 完整读取最终 PASS 计划和审查结论。
- 只在本地 repo 修改 tracked source。
- 不直接 SSH 修改服务器源码。
- 不执行旧目录归档、不启动 GPU 训练、不 push；这些由 Codex review 后执行。
- 可以运行本地测试。
- 不修改与计划无关的 dirty 文件。
- 每个阶段报告文件列表、测试、未解决风险。
- 完成后停止并等待 Codex review。

执行顺序：

1. centered wrapper + config semantics
2. train-only stats 工具
3. trainer/inference/evaluator 集成
4. checkpoint/resume/self-contained round-trip
5. 单测与 tiny smoke
6. archive/launch/watch 脚本（archive 默认 dry-run）
7. 文档与命令锁定

Codex 负责逐 diff review、退回修复、最终 push、服务器 pull、stats、烟测、归档与主训练监督。

---

## 14. Requirement-to-evidence matrix

| Requirement | 完成证据 |
|---|---|
| 7→15 日任务未漂移 | config sidecar + dataset shape test（centered 复用同一字段校验） |
| frozen mean 身份固定 | `.pth` SHA-256 + sidecar manifest SHA-256 + 每 rank 校验 |
| centered target 正确 | toy algebra test + real-batch diagnostic |
| anchor/mean 只加一次 | zero-innovation/reconstruction tests |
| stats 无泄漏 | stats JSON `split=train` + `indices_sha256` + 新 validator provenance 校验 + mean stats 一致性 |
| mean 真正冻结 | grad test + first-10-step runtime assertion |
| checkpoint 自包含 | 删除/隐藏原 mean 路径后的 round-trip inference test |
| legacy 路径未破坏 | 全量 golden/unit tests |
| resume 语义安全 | bare resume + conflict-fail + schema4/schema3 兼容 tests |
| 旧主训练可恢复 | archive manifest + SHA-256 before/after |
| 新输出替换规范位置 | archive 后 canonical path manifest |
| smoke 有效 | 20 step + resume + finite sample artifact |
| 正式烟测充分 | 500–1000 optimizer steps、`skipped_optimizer_steps==0`、有限 loss/梯度、checkpoint/resume 正常 |
| 双卡/单卡 batch 等价 | sidecar effective global batch=32 |
| 无 test 选择 | 进程/日志检查 + watcher config |
| 主训练正确运行 | tmux/PID/GPU/config/hash/前10步/skip 计数/首 checkpoint 证据 |

---

## 15. 当前计划外事项

以下工作不应阻塞今天主训练启动：

- 改造 IAFNO backbone
- 加 cross-attention/calendar features
- 联合微调 frozen mean
- 更换数据划分
- 重新访问 test 做模型选择
- 大规模超参数搜索
- 完整概率评估套件（CRPS/coverage 可在主训练启动后补齐）

这些事项必须另建 Phase 2 后续计划，不得混入本次关键路径。
