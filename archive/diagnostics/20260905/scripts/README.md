# archive/diagnostics/20260905/scripts

历史排障脚本备份；含旧路径或进程操作，勿直接执行。
本层文件用途见下方；子目录详见各自 README。历史内容仅供追溯，不代表当前执行指令。

## 本层项目树

```text
scripts/
├── fix_watcher_and_probe.sh  # 历史运维：修复当时的验证监测进程并探测状态
├── migrate_gpu03.sh  # 历史运维：等待指定 epoch 后迁移训练到 GPU 0 和 3
├── migrate_now.sh  # 历史运维：按当时配置迁移并恢复训练，不能直接用于当前实验
├── post_migrate_check.sh  # 历史运维：检查迁移后的训练和 GPU 状态
├── post_migrate_check2.sh  # 历史运维：补查迁移后进程、日志与训练状态
├── post_migrate_check3.sh  # 历史运维：再次核对迁移后的训练进度
├── pre_migrate_check.sh  # 历史运维：收集训练迁移前的进程与资源状态
├── probe_foreground.sh  # 历史运维：以前台方式检查当时的训练现场
├── probe_result.sh  # 历史运维：读取当时的探测结果
├── README.md  # 说明本目录用途并列出本层文件和子目录
├── server_contention_diag.sh  # 历史运维：检查服务器 CPU、内存和 GPU 资源争用
├── topo_check.sh  # 历史运维：检查 GPU 拓扑及通信相关环境
└── watcher_relocate.sh  # 历史运维：调整当时验证监测任务的运行位置
```
