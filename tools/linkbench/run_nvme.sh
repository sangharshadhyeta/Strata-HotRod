#!/bin/bash
# The storage and RAM links, measured one test at a time on a quiet machine. Read-only on the model file.
cd /root/.claude/jobs/743230e4/tmp/linkbench
F=/mnt/nvme/models/qwen38-flash-next/GSQ-RCO-IQ3_S/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
echo "NVMe link now: $(cat /sys/class/nvme/nvme0/device/current_link_speed) x$(cat /sys/class/nvme/nvme0/device/current_link_width); max $(cat /sys/class/nvme/nvme0/device/max_link_speed); read_ahead_kb $(cat /sys/block/nvme0n1/queue/read_ahead_kb)"
echo "--- sequential, O_DIRECT, 1 MiB"
./nvme_bench $F seq 6
echo "NVMe link after load: $(cat /sys/class/nvme/nvme0/device/current_link_speed) x$(cat /sys/class/nvme/nvme0/device/current_link_width)"
echo "--- random, O_DIRECT: expert-sized reads, by reads in flight"
for c in 2048 896 512 128; do
  for t in 1 4 16 32; do ./nvme_bench $F rand 4 $t $c; done
done
echo "--- mmap random reads of one 896 KiB expert slice (cold file): kernel read-around (normal) against MADV_RANDOM"
for t in 1 8; do
  ./nvme_bench $F mmap 8 $t 896 0
  ./nvme_bench $F mmap 8 $t 896 1
done
echo "--- RAM memcpy"
./nvme_bench - ram 3 1
./nvme_bench - ram 3 6
./nvme_bench - ram 3 12
echo "DONE"
