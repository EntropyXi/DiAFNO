# 项目导航与保留规则

当前有旧主训练和时空消融两套 worktree；目录同名的模块可能属于不同版本，不能直接互相覆盖。
这次只整理文档、归档历史计划和排障脚本、添加文件头用途注释，不改变模型、数据协议或训练参数。

## 目录与入口

| 目录/入口 | 用途 |
|---|---|
| `diafno/` | 正式数据、模型、训练、验证与推理实现。 |
| `deterministic_iafno/` | 确定性残差基线、冻结均值的 centered 扩散及统计量校验。 |
| `trainer_ostia.py` | 开始或恢复训练。 |
| `validate_ostia.py` | 加载 checkpoint 进行在线验证/测试评分。 |
| `infer_ostia.py` | 批量预测并保存结果。 |
| `evaluate_ostia.py` | 对已经保存的预测文件离线评分，不直接加载 checkpoint。 |
| `configs/` | 运行配置；包含 plan 字样的机器调度 JSON 仍是配置，不按文字计划书移走。 |
| `scripts/` | 显存探测、烟测、启动、监测、消融与图表工具；不会因整理自动执行。 |
| `archive/plans/20260905/` | 按原相对路径备份计划、历史运行手册、执行提示词和评审要求。 |
| `archive/diagnostics/20260905/` | 若存在，保存临时诊断与迁卡工具；部分会操作旧进程，不应直接执行。 |
| `artifacts/` | 审计、manifest、统计量、调度状态及代表性烟测。 |
| `experiments/` | 实验权重、曲线、逐样本累计量和评估报告，保持路径兼容。 |

## 实验保留范围

保留代表性 `artifacts/smoke/`、centered 结构烟测、A0–A5 的 stage1/2/3、A5 finetune、旧主训练 best/latest/epoch 权重及其 sidecar、归一化统计、冻结均值身份、test-1000/test-200 报告、配对 bootstrap 累计量和最终展示图。旧失败实验本次也不盲删，以保留诊断链；这次目标是可读性而不是释放权重空间。

`three_method_layout_demo` 是合成排版演示，不是测试结果。最终单区域图目录顶层的累计量属于单案例，200 样本来源以 `aggregate_source/` 或原始 `three_method_ep22_test200_e16_20260905/` 为准。

## 两套 worktree

- 旧主训练：本地 `DiAFNO`，服务器 `/data2/user/zzx/exam_preprocessed/DiAFNO`，分支 `OSTIA_SST`。
- 时空消融：本地 `DiAFNO-spatiotemporal-ablation`，服务器 `/data2/user/zzx/exam_preprocessed/DiAFNO_spatiotemporal_ablation`，分支 `codex/ostia-spatiotemporal-ablation`。
- 早期 `DiAFNO_deterministic_baseline` 是独立历史 worktree，本次不改写其历史代码。

每层 README 列出本层文件的一句话用途；子目录自己的 README 继续展开。树合并本地和服务器已有产物，标记“服务器产物”的项目本地可能不存在。Git 内部目录、虚拟环境和 Python 缓存不属于业务目录，不添加 README。

## 备份和恢复

整理前的本地源码快照在同级 `DiAFNO_readability_backup_20260905/`，服务器快照在 `/data2/user/zzx/exam_preprocessed/DiAFNO_readability_backup_20260905/`。快照保留整理前的未提交内容，不包括庞大实验权重；实验权重本次不改动。

计划书的新路径为 `archive/plans/20260905/<原相对路径>`。原位置已移出，恢复时从这里复制指定文件即可，不要把整个备份盲目覆盖回项目。归档正文保持原样，其中相对链接或旧命令可能指向当时布局，查找配套文件以原相对路径和本导航为准。
