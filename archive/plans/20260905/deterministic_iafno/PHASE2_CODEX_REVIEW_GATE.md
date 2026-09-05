# Phase 2 Codex 实现验收门禁

DeepSeek 执行对话完成后，Codex 必须逐项收集权威证据；任一项缺失均退回修复，不得 push。

## A. 范围与工作树

- [ ] 修改只覆盖最终 PASS 计划允许的文件。
- [ ] 用户原有 `STATUS.md`、scratch、迁移/诊断脚本等无关 dirty 文件未被覆盖。
- [ ] 无删除、无服务器直接源码编辑、无隐式归档或训练启动。
- [ ] `git diff --check` 通过。

## B. 数学目标空间

- [ ] frozen mean 使用 `predict()`，不使用 `_network_prediction`，不重复应用 mean lead stats。
- [ ] `e = residual_target - mu` 与 `z = (e-m)/s` 在 fp32 执行。
- [ ] sample 只重建一次 `mu + m + s*z_hat`；anchor 只由 evaluator 加一次。
- [ ] zero-innovation、real-batch reconstruction、transform round-trip 测试通过。

## C. 冻结、优化器、AMP 与 DDP

- [ ] `mean_model.requires_grad_(False)` 且 wrapper `.train()` 后 mean 仍 eval。
- [ ] optimizer 参数集合严格等于 trainable diffusion 参数。
- [ ] DDP 下无 mean grad/unused-param 静默问题。
- [ ] overflow skip 不推进 optimizer/scheduler；skip count checkpoint/history 可恢复。
- [ ] sampler 属性读写双向委托测试通过。

## D. checkpoint、resume 与推理

- [ ] schema 4 immutable fields 完整；schema 3 legacy 兼容测试通过。
- [ ] centered checkpoint state dict 含 frozen mean 权重。
- [ ] checkpoint config/sidecar 含 mean stats、innovation stats 与双 SHA 身份。
- [ ] 隐藏原 mean/stats 路径后 inference 与 bare resume round-trip 通过。
- [ ] 冲突 override、错误 hash、错误 stats provenance 均 fail closed。

## E. 数据与统计

- [ ] centered stats 只读 train split，index 选择确定且 chunk-aware。
- [ ] JSON 含 dataset size、indices SHA、mean SHA、mean semantics SHA、mean/innovation stats。
- [ ] validator 拒绝 val/test 元数据、错误 target space、错误 lead 数量、非有限或非正 std。
- [ ] 相同命令确定性复算通过。

## F. legacy 回归与完整测试

- [ ] legacy deterministic/diffusion golden tests 数值不变。
- [ ] 完整 `deterministic_iafno/tests` 通过。
- [ ] compileall、import、config schema 与 tiny CPU smoke 通过。
- [ ] 测试覆盖与最终 PASS requirement matrix 一一对应。

## G. 运维脚本

- [ ] archive 脚本默认 dry-run，realpath/symlink/active-writer/target-exists 检查 fail closed。
- [ ] smoke 使用独立目录，不污染 canonical/archive。
- [ ] launcher 只读权威 JSON，并打印 config/mean/stats hash。
- [ ] 2-GPU 与 1-GPU fallback 都保持 effective global batch 32。
- [ ] watcher 在 epoch 5 消融后锁定推理配置，不用 test 选择 checkpoint。

只有 A–G 全部有直接证据时，Codex 才可提交、push，并在服务器执行 `git pull --ff-only`。
