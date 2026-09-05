#!/bin/bash
# 用途：历史运维：以前台方式检查当时的训练现场。
echo "=== probe process state ==="
pgrep -af load_probe.py | grep -v grep || echo "probe dead (killed on ssh close)"
echo "=== probe log ==="
cat /tmp/ostia_ft_logs/load_probe.log 2>/dev/null | tail -5
echo "=== run probe in foreground (max 240s) ==="
timeout 280 /data2/user/zzx/ENTER/envs/DiAFNO/bin/python -u /tmp/ostia_ft_logs/load_probe.py 2>&1 | tail -5
echo PROBE_DONE
