# result

三方法对比（DiAFNO 集合 16 / 确定性 IAFNO / persistence，test-200）的对外展示产物快照；
源数据为服务器 `experiments/three_method_ep22_test200_e16_20260905/`。

## 本层项目树

```text
result/
├── REPORT.md  # 按 forecast 日给出 RMSE/MSE/CRPS/skill 及 22 日配对分块 bootstrap 95% CI 的三方法对比报告
├── forecast_region_000.pdf  # 000 号代表区域的 Day 1/5/10/15 三方法预报面板图（矢量版）
└── forecast_region_000.png  # 000 号代表区域的 Day 1/5/10/15 三方法预报面板图（位图版）
```
