#!/bin/bash
# Comprehensive GPU and VLLM cleanup script
# Usage: bash suze_experiments/20251016/cleanup_gpu.sh [port]

PORT="${1:-6000}"

echo "=== Cleaning up VLLM and GPU processes ==="

# Kill all VLLM-related processes with multiple patterns
echo "Killing VLLM processes..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "vllm complete" 2>/dev/null || true

# Kill load balancers
echo "Killing load balancers..."
pkill -9 -f "load_balancer" 2>/dev/null || true

# Kill any process using the GPUs
echo "Killing GPU processes..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
if [ -n "$GPU_PIDS" ]; then
    echo "$GPU_PIDS" | xargs -r kill -9 2>/dev/null || true
    echo "Killed GPU processes: $GPU_PIDS"
else
    echo "No GPU processes found"
fi

# Clean up ports
echo "Cleaning up ports around $PORT..."
for p in $PORT $((PORT+1)) $((PORT+2)) $((PORT+3)) $((PORT+4)); do
    PORT_PIDS=$(lsof -ti:$p 2>/dev/null)
    if [ -n "$PORT_PIDS" ]; then
        echo "Killing processes on port $p: $PORT_PIDS"
        lsof -ti:$p | xargs -r kill -9 2>/dev/null || true
    fi
done

sleep 2

# Verify cleanup
echo ""
echo "=== Verification ==="
nvidia-smi

echo ""
echo "=== Port status ==="
for p in $PORT $((PORT+1)) $((PORT+2)) $((PORT+3)) $((PORT+4)); do
    if lsof -ti:$p >/dev/null 2>&1; then
        echo "WARNING: Port $p still in use"
    else
        echo "Port $p is free"
    fi
done

echo ""
echo "Cleanup complete!"

