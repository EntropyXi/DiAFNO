# DeepSeek-V4-Pro Max 双高精度计划审查提示词

你现在是 DiAFNO 项目的独立高级研究工程师与红队审查员。

本轮只审查计划，不写代码、不修改文件、不运行训练、不 push、不操作 SSH 服务器。
请在 DiAFNO 工作区完整读取以下文件后再输出结论：

1. `deterministic_iafno/PHASE2_MAIN_TRAINING_PLAN.md`
2. `deterministic_iafno/RUNBOOK.md`
3. `deterministic_iafno/reports/PHASE0_PHASE1_FINAL_REPORT_ZH.md`
4. 与计划直接相关的当前 config、trainer、model、evaluator、checkpoint semantics 源码

任务的权威定义是**连续 7 日 SST 输入预测随后 15 日 SST**。任何旧的“7周→15周”、
weekly preprocessing 或月尺度文档都属于过期上下文，不得据此改变本计划的日尺度目标。

已冻结的 deterministic mean 是：

- `experiments/det_lead_standardized/epoch_015.pth`
- SHA-256：`cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6`

Phase 1 paired block-bootstrap gate 已通过。本计划目标是在 12 小时窗口内完成 centered
diffusion 的正确实现、500–1000 optimizer-step 烟测、旧主训练可恢复归档，并让新主训练
在 tmux 中正式稳定运行。主训练默认双卡，没有可用双卡时允许单卡，但不得终止或抢占其他
用户进程。

请严格执行两遍互相独立的高精度审查。

## Pass A：正向架构与可执行性审查

逐项验证：

1. 数学目标空间是否严格为：
   - normalized future residual `r = target - anchor`
   - frozen mean `mu(c)`
   - centered innovation `e = r - mu(c)`
   - train-only per-lead standardized innovation `z = (e-m)/s`
   - sample 重建 `anchor + mu + m + s*z`
2. anchor、mean、normalization 是否可能被重复应用或遗漏。
3. frozen mean 是否能在 DDP、`.train()`、AMP 下保持 eval/no-grad。
4. centered checkpoint 是否真正自包含，resume/inference 是否不依赖原 mean 路径。
5. config、sidecar、schema、hash 是否足以 fail closed。
6. train-only stats 是否存在 val/test 泄漏或采样偏差。
7. optimizer、scheduler、GradScaler、gradient accumulation、effective batch 是否一致。
8. 旧 diffusion/deterministic 路径能否保持 golden compatibility。
9. 旧目录归档、新 canonical output 替换、回滚是否可执行且不覆盖。
10. 500–1000 step 烟测是否足以捕获目标空间、resume、采样和内存错误。
11. 双卡与单卡 fallback 是否保持相同有效 batch、LR 和数据暴露量。
12. 12 小时时间预算是否现实，哪些步骤是真正关键路径。

## Pass B：对抗性失败审查

假设实现者会犯最危险但表面不报错的错误，主动寻找：

1. deterministic mean 输出已反标准化却被再次应用 lead stats。
2. centered diffusion sample 已加 mean，evaluator 又加一次。
3. wrapper 返回绝对 SST，evaluator 再加 anchor。
4. mean 权重进入 optimizer 或被 wrapper `.train()` 切回 train mode。
5. centered stats 来自错误 split、错误 mask、错误 checkpoint 或错误样本索引。
6. sigma_data 与 standardized innovation 尺度不一致。
7. checkpoint fresh/init/resume 三条路径行为不一致。
8. sidecar 路径正确但内容 hash 不匹配。
9. DDP 两个 rank 加载不同 mean 或取到重复样本。
10. AMP overflow 被误记为正常 optimizer step。
11. 归档脚本对错误目录、symlink、已存在目标或活动写进程处理不安全。
12. 新训练误写到 archive、旧目录或 deterministic mean 目录。
13. smoke 使用了与正式训练不同的 target/stat/sigma/LR，导致假通过。
14. watcher 偷看 test 或用 test 选择 checkpoint。
15. GPU 瞬时 0% 被误判为空闲并与他人训练冲突。
16. 计划看似完整，但 requirement-to-evidence matrix 中证据不足。

## 固定参数约束

主训练前不做大规模 EDM 扫描。除非你给出 blocking 级数学/实现证据，否则保持：

- `sigma_data=1.0`
- `P_mean=-1.2`
- `P_std=1.2`
- `rho=7`
- `sigma_min=0.002`
- `sigma_max=80`
- 主训练 35 epoch、31200 samples/epoch、effective global batch=32

epoch 5 后才做廉价推理消融：steps 16 vs 32、ensemble 1 vs 4，`S_churn=0` 为主。

## 输出格式

必须按以下格式输出，不得只给泛泛建议：

1. 第一行：`VERDICT: PASS` 或 `VERDICT: REVISE`
2. `Pass A findings`
3. `Pass B findings`
4. `Blocking issues`
   - 每项给：编号、计划章节/源码证据、失败机制、最小充分修正
5. `Non-blocking improvements`
6. `Required replacement text`
   - 给出可直接替换计划相应段落的完整中文文本
7. `Requirement-to-evidence audit`
   - 每个 requirement 标为 PROVEN / PARTIAL / MISSING / CONTRADICTED
8. `12-hour critical path verdict`
9. `Final re-review checklist`

如果 verdict 为 REVISE，等待 Codex 修改计划后再次审查；只有你明确输出
`VERDICT: PASS` 才允许创建新的执行对话。
