#!/bin/bash
cd /root/.claude/jobs/743230e4/tmp/linkbench
F=/mnt/nvme/models/qwen38-flash-next/GSQ-RCO-IQ3_S/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
echo "--- RAM <-> VRAM"
./pcie_bench
echo "--- chain: NVMe (O_DIRECT random) -> pinned RAM -> VRAM"
for t in 1 4 8; do ./pcie_bench $F $t 896; done
./pcie_bench $F 4 2048
echo DONE
