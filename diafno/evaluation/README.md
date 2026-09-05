# diafno/evaluation

在线验证、离线评估、配对统计与对比图。
本层文件用途见下方；子目录详见各自 README。实验产物保持原路径；本次不改变训练或评估行为。

## 本层项目树

```text
evaluation/
├── __init__.py  # 声明该 Python 包并组织其公共接口
├── bootstrap.py  # 对配对预报误差执行时间分块 bootstrap 并计算 skill 区间
├── config.py  # 解析在线验证的样本、采样、设备及输出配置
├── evaluator.py  # 读取推理文件，累计并输出离线 SST 指标
├── main.py  # 提供已保存预测文件的离线评估命令入口
├── method_comparison.py  # 计算三方法配对指标并绘制三行四列 SST 图与 Markdown 表
├── metrics.py  # 累计有效像素的 SST 误差、相关系数及 persistence skill
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── validation_main.py  # 连接在线验证参数与 checkpoint 验证器
└── validator.py  # 执行模型或 persistence 预测、条件消融及分 lead 评分
```
