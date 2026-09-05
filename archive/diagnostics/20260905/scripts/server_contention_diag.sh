#!/bin/bash
# 用途：历史运维：检查服务器 CPU、内存和 GPU 资源争用。
# Server contention diagnostic for the OSTIA training box (8 GPUs, 8 NUMA nodes).
# Read-only; safe to run while training. Pipe to ssh, e.g.:
#   Get-Content scripts/server_contention_diag.sh -Raw | ssh 8.138.27.9 "bash -s"
echo "=== time ==="
date -u +"%Y-%m-%d %H:%M:%S UTC"
echo "=== our trainer processes (states; D=blocked IO) ==="
ps -u zzx -o pid,state,pcpu,pmem,etime,cmd | grep -E "trainer_ostia|pt_data_worker" | grep -v grep | head -12
echo "=== full GPU board (idx, mem used/total MiB, util%) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== compute apps on GPUs (all users) ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -20
echo "=== load / nproc ==="
nproc; cat /proc/loadavg
echo "=== global memory ==="
free -g | head -2
echo "=== memory + io pressure ==="
head -1 /proc/pressure/memory
head -1 /proc/pressure/io 2>/dev/null || echo "no pressure/io"
echo "=== per-NUMA-node free GiB ==="
for N in /sys/devices/system/node/node*/meminfo; do
  echo "$N: $(grep MemFree $N | awk '{printf "%.1f", $4/1048576}') GiB free"
done
echo "=== top 10 RSS consumers ==="
ps -eo user,pid,rss,pcpu,etime,comm --sort=-rss | head -11 | awk '{printf "%-8s pid=%-8s rss=%.1fGB cpu=%s%% et=%s %s\n", $1, $2, $3/1048576, $4, $5, $6}'
echo "=== per-user CPU sum / proc count ==="
ps -eo user,pcpu --no-headers | awk '{u[$1]+=$2; n[$1]++} END {for (x in u) printf "%-10s procs=%d cpu_sum=%d\n", x, n[x], u[x]}' | sort -k3 -rn | head -10
echo "=== NUMA placement of our ranks ==="
for P in $(pgrep -f "trainer_ostia.py" | head -2); do
  echo "--- rank pid $P ---"
  grep -oE "N[0-9]+=[0-9]+" /proc/$P/numa_maps 2>/dev/null | awk -F'[= ]' '{s[$1]+=$2} END {for (k in s) printf "%s=%.1fGiB ", k, s[k]*4/1048576; print ""}'
done
echo "=== iostat snapshot ==="
iostat -x 1 1 2>/dev/null | grep -E "Device|nvme|sd" | tail -4 || echo "iostat n/a"
echo DIAG_DONE
