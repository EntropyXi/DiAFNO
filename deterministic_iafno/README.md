# deterministic_iafno

确定性残差基线与冻结均值的 centered diffusion 扩展。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
deterministic_iafno/
├── reports/  # 确定性基线和诊断阶段的结论与机器可读汇总
├── tests/  # 确定性和 centered 模型、统计量及恢复训练的回归测试
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── centered_diffusion.py  # 冻结确定性均值网络，并训练和采样中心化创新扩散模型
├── centered_stats.py  # 校验创新统计量、冻结均值身份及其来源一致性
├── checkpoint_semantics.py  # 定义和校验 checkpoint 的模型、数据及续训语义
├── compute_centered_stats.py  # 只用训练集计算相对冻结均值的逐 lead 创新统计量
├── compute_lead_stats.py  # 只用训练集计算未来 SST 残差的逐 lead 均值与标准差
├── losses.py  # 计算有效像素掩膜下、适配 DDP 全局归一化的 MSE
├── model.py  # 封装确定性 IAFNO，预测按 lead 标准化的 SST 残差
├── PHASE2_IMPLEMENTATION_REPORT.md  # 记录 centered diffusion 的实际实现与验收结果
├── README.md  # 说明本目录用途并列出本层文件和子目录
└── STATUS.md  # 保留该分支原有阶段状态记录，不自动改写历史结论
```

## 原有使用说明（保留）

# Deterministic IAFNO work area

This directory isolates the small-change investigation for the OSTIA
7-day-to-15-day task.

Execution order is intentionally fixed:

1. Freeze and verify the legacy evaluator contract.
2. Add fail-fast checkpoint/resume semantics.
3. Add persistence/trend and clean condition ablations.
4. Add the raw-backbone deterministic IAFNO path.
5. Compare raw-residual MSE with train-only lead-standardized MSE.
6. Stop if the deterministic model does not beat persistence on validation.
7. Only after that gate, consider frozen-mean centered diffusion.

The current implementation phase covers items 1-5 plus the Phase 2
frozen-mean centered diffusion:

- `centered_diffusion.py` — `FrozenMeanCenteredDiffusion` wrapper
  (frozen eval mean, fp32 centered innovation, single reconstruction,
  two-way sampler attribute delegation);
- `centered_stats.py` / `compute_centered_stats.py` — train-only
  centered innovation statistics with fail-closed provenance
  validation;
- `configs/ostia_centered_diffusion_main.json` — the authoritative main
  training config (7-day → 15-day, sigma_data=1.0);
- `scripts/archive_legacy_ostia_main.sh` (default dry-run),
  `init_ostia_centered_main.sh`, `run_ostia_centered_main.sh`,
  `smoke_ostia_centered.sh`,
  `watch_ostia_centered.sh`.

See [已归档的 PHASE2_CENTERED_RUNBOOK](../archive/plans/20260905/deterministic_iafno/PHASE2_CENTERED_RUNBOOK.md) for the locked algebra and the
reproducible server command sequence.  Server-side archiving, GPU
smoke and the main training launch are Codex-only steps after review.
