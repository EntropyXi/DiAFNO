# Phase 2 centered diffusion 执行提示词

你是 DiAFNO 项目的实现 agent。开始前必须完整读取：

1. `deterministic_iafno/PHASE2_MAIN_TRAINING_PLAN.md`
2. `deterministic_iafno/PHASE2_PLAN_REVIEW_PROMPT.md`
3. `deterministic_iafno/PHASE2_PLAN_REREVIEW_PROMPT.md`
4. 最终 `VERDICT: PASS` 的复审结论
5. `deterministic_iafno/RUNBOOK.md`
6. `deterministic_iafno/reports/PHASE0_PHASE1_FINAL_REPORT_ZH.md`
7. 与 config、trainer、model、inference、evaluation、checkpoint semantics 直接相关的当前源码与测试

权威任务是连续 7 日 SST 输入预测随后 15 日 SST。任何 weekly、monthly、7周→15周材料均为过期
上下文，不得改变日尺度任务。

## 操作边界

- 只修改本地仓库的必要 tracked source、tests、configs、scripts 与文档。
- 不操作 SSH 服务器，不 push，不归档旧目录，不启动服务器 GPU 训练。
- 不删除任何文件，不修改与任务无关的 dirty/untracked 文件。
- 所有 shell 命令使用 Git Bash；非必要不使用 PowerShell。
- 所有实现必须以最终 PASS 计划为准，不得自行缩小功能范围或改成更容易测试的替代方案。
- archive 脚本必须默认 dry-run；执行归档只能由 Codex 在服务器完成。

## 强制实现顺序

1. centered model/config/checkpoint schema 与 frozen mean wrapper。
2. train-only centered innovation stats 工具与 provenance validator。
3. trainer、AMP skip、scheduler、DDP 与 resume 集成。
4. inference/evaluator 自包含重建与 sampler 属性双向委托。
5. 完整单元、golden、round-trip、resume 与 tiny CPU smoke tests。
6. 配置 JSON、archive/launch/smoke/watch 脚本。
7. 文档和可复现命令。

## 不可妥协的数学与运行契约

- `mu = frozen_mean.predict(condition)` 已处于 normalized residual 空间。
- `e = residual_target - mu`，`z = (e-m)/s` 必须在 fp32 计算。
- sampler 重建只执行一次 `r_hat = mu + m + s*z_hat`；wrapper 不加 SST anchor。
- zero innovation 必须精确退化为 deterministic mean。
- mean 永久 eval、无梯度；optimizer 只包含 trainable diffusion 参数。
- `S_churn/sigma_min/sigma_max/rho/num_sample_steps` 必须读写双向委托。
- centered checkpoint 必须自包含 mean 权重、mean stats、innovation stats 与不可变语义；隐藏原 mean
  路径后仍可推理和 resume。
- AMP overflow skip 不推进 optimizer 或 scheduler；记录 skip count 与发生 step。
- fresh run 的 mean `.pth` SHA、sidecar immutable SHA、mean stats、train-only stats provenance 必须
  每 rank fail closed 校验。
- schema 4 centered 语义与 schema 3 legacy 只读/恢复兼容均需测试。
- `configs/ostia_centered_diffusion_main.json` 是唯一权威主训练配置；centered 模式拒绝
  `sigma_data != 1.0`。

## 工作与报告协议

每完成一个阶段：

1. 列出修改文件与原因。
2. 展示相关测试命令、结果与覆盖的计划 requirement。
3. 运行 `git diff --check`。
4. 说明未解决风险，不得用“看起来没问题”代替证据。

最终必须：

- 运行完整 `deterministic_iafno/tests`，不能只跑新测试。
- 运行 compile/import/config schema 检查。
- 展示 archive 脚本 dry-run，不能执行 archive。
- 给出逐项 requirement-to-evidence matrix。
- 不提交、不 push；停止并等待 Codex review。

