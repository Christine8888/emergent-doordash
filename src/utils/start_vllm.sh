#!/bin/bash

set -e
set -o pipefail

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <model_path> [tensor_parallel] [model_name] [n_devices]"
    echo "Examples:"
    echo "  $0 /workspace/model"
    echo "  $0 /workspace/model 2 my-model 4"
    exit 1
fi

MODEL_PATH="$1"
TP="${2:-4}"
MODEL_NAME="${3:-}"
N_DEVICES="${4:-$TP}"

export HF_HOME="/workspace/.cache/huggingface"

VLLM_PIDS=()
NGINX_PID=""
SHUTTING_DOWN=false

cleanup() {
    SHUTTING_DOWN=true
    echo "Shutting down servers..."

    [ -n "$NGINX_PID" ] && kill "$NGINX_PID" 2>/dev/null || true
    sudo nginx -s quit 2>/dev/null || true

    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    sleep 5
    pkill -f "vllm serve" 2>/dev/null || true

    for port in {9000..9004}; do
        lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done

    rm -f /tmp/vllm_nginx_$$.conf
    echo "Shutdown complete"
    exit 0
}

trap cleanup EXIT INT TERM

if [ $((N_DEVICES % TP)) -ne 0 ]; then
    echo "Error: n_devices ($N_DEVICES) must be divisible by tensor_parallel ($TP)"
    exit 1
fi

NUM_INSTANCES=$((N_DEVICES / TP))

VLLM_ARGS=(
    --dtype auto
    --max-model-len 32768
    --tensor-parallel-size $TP
    --enable-prefix-caching
    --max-num-seqs 32
    --max-num-batched-tokens 131072
    --enable-chunked-prefill
    --gpu-memory-utilization 0.9
    --kv-cache-dtype auto
    --max-parallel-loading-workers 2
)

[ -n "$MODEL_NAME" ] && VLLM_ARGS+=(--served-model-name "$MODEL_NAME")

echo "Starting $NUM_INSTANCES vLLM instance(s)"
echo "Model: $MODEL_PATH"
echo "TP: $TP | Devices: $N_DEVICES"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((9001 + i))
    GPU_START=$((i * TP))
    GPU_END=$((GPU_START + TP - 1))

    CUDA_DEVICES=$(seq $GPU_START $GPU_END | tr '\n' ',' | sed 's/,$//')

    echo "Starting instance $((i + 1)) on GPU(s) $CUDA_DEVICES (port $PORT)"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" --port $PORT &
    VLLM_PIDS+=($!)
    sleep 5
done

echo "Waiting for servers..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((9001 + i))
    while ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; do
        [ "$SHUTTING_DOWN" = true ] && exit 0
        sleep 2
    done
    echo "Server on port $PORT ready"
done

NGINX_CONFIG="/tmp/vllm_nginx_$$.conf"
cat > "$NGINX_CONFIG" << EOF
events { worker_connections 1024; }
http {
    upstream vllm_backend { least_conn;
EOF

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    echo "        server localhost:$((9001 + i));" >> "$NGINX_CONFIG"
done

cat >> "$NGINX_CONFIG" << 'EOF'
    }
    server {
        listen 9000;
        client_max_body_size 100M;
        location / {
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            proxy_buffering off;
        }
    }
}
EOF

sudo nginx -c "$NGINX_CONFIG" &
NGINX_PID=$!

echo "All servers started"
echo "Load balancer: http://localhost:9000"
echo "Press Ctrl+C to stop"

wait
