# 三种方法的测试图与指标表

`scripts/compare_ostia_methods.py` 一次配对评估三个方法：

1. DiAFNO 主训练（普通 diffusion 或 centered diffusion）：图中用集合均值；
2. IAFNO 确定性预测；
3. persistence：最后一个输入日 SST，重复到未来 15 天。

每张图固定同一区域、同一初始化时刻。**三行是三个方法，四列是 Day 1/5/10/15 的预测 SST。**
所有图共享一个 SST 色标，保留数据的像素方向，用灰色标注对应预测日的无效目标像素。
不加真实值行、误差列或置信度列。另存目标像素数组便于后续审计。

## 运行

在服务器仓库根目录运行，下列模型/数据路径需要替换成真实路径。
只对已经冻结的主训练与 baseline 做 test；尚在架构筛选时用 `--split val`。

```bash
CUDA_VISIBLE_DEVICES=4 python -u scripts/compare_ostia_methods.py --diafno-checkpoint /path/to/diafno/best_model.pth --iafno-checkpoint /path/to/iafno/best_model.pth --h5-path /path/to/ocean_temperature_data_patched.h5 --data-manifest artifacts/ostia_data_manifest_real.json --output-dir experiments/three_method_test_200 --split test --max-samples 200 --plot-samples 3 --ensemble-members 16 --sampling-steps 16 --bootstrap-replicates 2000 --sst-unit K
```

先用 `--max-samples 3 --ensemble-members 2 --bootstrap-replicates 0` 和新输出目录核对流程，
正式报告再使用固定的 200/1000 样本与 16 成员。推理成本随样本数、成员数和采样步数增长。
该脚本每次处理一个样本，在同一 GPU 顺序运行两种模型；集合成员及时转 CPU，避免把所有成员留在 GPU。
默认每种模型各有 2 个数据加载 worker，可以通过 `--num-workers 0` 减少 CPU 内存开销。

脚本复用现有验证器的 checkpoint 加载、标准化检查、模型预测和残差还原逻辑，
在 CPU 上计算经验集合 CRPS。它不修改训练，也不从旧的汇总 RMSE 反推 CRPS。
模型和数据契约必须兼容；跨模型时间或 mask 不一致会明确报错，不自动忽略。
旧模型未绑定日期 manifest 时可以按其契约导出；有绑定时必须给匹配的文件。
同次对比共用一份 manifest，以确保固定 seed 对应同一组物理样本。

SST 单位默认是 `source units`。只有确认数据单位后才设 `--sst-unit K` 或 `--sst-unit degC`。
若源数据为 K、希望图表用摄氏度，加 `--sst-unit degC --sst-offset -273.15`。
标签本身不改变数据；不得对已经是摄氏度的数据再次减 273.15。

## 指标定义

Markdown 报告按 Day 1、5、10、15、overall 分为五张表，每张表三行。

| 指标 | 计算口径 |
|---|---|
| RMSE / MSE / MAE / bias / corr | DiAFNO 集合均值、IAFNO 单点和 persistence 单点；相同有效像素 |
| CRPS | E\|X−y\| − ½ E\|X−X′\|，经验集合版本（非 fair CRPS） |
| 确定性 CRPS | 将预测视作单点分布，因此等于 MAE |
| MSE skill | 1 − MSE(method) / MSE(persistence) |
| CRPS skill | 1 − CRPS(method) / CRPS(persistence) |
| skill 95% CI | 按输入初始化时间划分 22 日块，配对 bootstrap；方法和空间样本保持配对 |
| overall | 所有 15 个预测日、所有评估样本的有效像素加权汇总 |

若 persistence 分母为 0，skill 未定义，记为 `—`。少于两个时间块或关闭 bootstrap 时 CI 记为 `—`。
CRPS skill 的参照是单点 persistence；这与另行定义的气候态或概率 persistence 不是同一基准。
固定成员数后再比较不同 DiAFNO checkpoint，避免有限集合大小改变经验 CRPS 的比较口径。
raw SST corr 不是去气候态 ACC。小样本结果与区间仅用于核对流程，不能充当完整测试结论。

## 输出

- `REPORT.md`：五张指标表、预测图引用和完整来源记录；
- `forecast_region_000.png/.pdf` 等：每区域一张 3×4 图；
- `comparison.json`：全部数值与指标设置；
- `run_manifest.json`：开始评估前固定的 checkpoint SHA256、样本索引、种子、成员数等；
- `paired_score_sums.npz`：每样本每预测日的 SSE、CRPS 和有效像素计数，用于复核区间；
- `case_*.npz`：展示区域的三种预测均值、target、mask、时间与区域标识。

表格使用全部 `--max-samples` 个配对样本；图使用其中预先等间隔选定的 `--plot-samples` 个。
选图位置在模型推理前决定，不按照误差或视觉效果挑最好看的样本。
输出目录非空时拒绝运行，防止把不同 checkpoint 的结果混在一起。

已有输出重新排版，无须 GPU 或再次推理：

```bash
python scripts/compare_ostia_methods.py --output-dir experiments/three_method_test_200 --render-only --dpi 300
```

重画会写入该目录下新的 `render_时间戳` 子目录，原始结果保留。
