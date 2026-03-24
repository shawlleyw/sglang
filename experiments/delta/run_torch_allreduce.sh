#!/bin/bash
# Usage: ./run_torch_allreduce.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/miniconda3/envs/sglang/bin/python"

JOBID=${SLURM_JOB_ID:-$(squeue -u "$USER" -h -o '%i' | head -1)}
NODELIST=$(squeue -j "$JOBID" -o '%N' -h)
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
NNODES=${#NODES[@]}
MASTER=${NODES[0]}
PORT=29500

echo "Job $JOBID: $((NNODES * 4)) GPUs on $NNODES nodes ($NODELIST)"
echo "Master: $MASTER:$PORT"
echo ""

PIDS=()
for i in "${!NODES[@]}"; do
    ssh "${NODES[$i]}" "NCCL_DEBUG=WARN \
        MASTER_ADDR=$MASTER MASTER_PORT=$PORT \
        $PYTHON -m torch.distributed.run \
            --nproc_per_node=4 --nnodes=$NNODES --node_rank=$i \
            --master_addr=$MASTER --master_port=$PORT \
            $SCRIPT_DIR/torch_allreduce.py" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done
