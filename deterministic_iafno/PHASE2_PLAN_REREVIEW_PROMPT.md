# Phase 2 主训练计划复审提示词

本轮仍然只审查计划：不得写代码、修改文件、运行训练、push 或操作 SSH 服务器。

请重新完整读取：

1. `deterministic_iafno/PHASE2_MAIN_TRAINING_PLAN.md`
2. `deterministic_iafno/PHASE2_PLAN_REVIEW_PROMPT.md`
3. 第一轮审查结论与 R1–R12 replacement text
4. 与第一轮五个 blocking issue 直接相关的当前源码

请先逐项确认第一轮五个阻塞项是否已经被修订版充分消除：

- B-1：§6.3 与 §11 的正式烟测门槛一致，均为 500–1000 optimizer steps，默认 1000，绝不小于 500。
- B-2：冻结 mean 的 `.pth`、sidecar immutable semantics、mean lead stats 都有 fail-closed 身份固定和每-rank 校验。
- B-3：wrapper 的 sampler 属性为读写双向委托，尤其 `S_churn` 不会被 wrapper 静默遮蔽。
- B-4：AMP overflow skip 不推进 scheduler；烟测要求 skip count 为 0，主训练记录并审查 skip rate。
- B-5：centered stats 有 train-only provenance 硬校验，数值近似恒等式不再被当作无泄漏证据。

然后重新独立执行 Pass A（正向架构与可执行性）和 Pass B（对抗性失败审查）。权威任务仍是连续
7 日 SST 输入预测随后 15 日 SST；weekly、monthly、7周→15周材料均为过期上下文。

输出必须以以下之一开头：

- `VERDICT: PASS`
- `VERDICT: REVISE`

若为 `REVISE`，只列仍然阻塞正确实现或正确主训练启动的问题，并给出证据、失败机制和可直接替换
计划段落的完整文本，然后停止等待 Codex 修订。若为 `PASS`，给出 requirement-to-evidence matrix，
明确说明可以进入独立的新执行对话；本对话仍不得进入实现。
