#!/bin/bash

set -e
set -o pipefail

if [ $# -lt 1 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 <model_path> [tensor_parallel] [model_name] [n_devices] [base_port]"
    echo "Examples:"
    echo "  $0 /workspace/model"
    echo "  $0 /workspace/model 2 my-model 4 9000"
    exit 1
fi

MODEL_PATH="$1"
TP="${2:-4}"
MODEL_NAME="${3:-}"
N_DEVICES="${4:-$TP}"
BASE_PORT="${5:-9000}"

export HF_HOME="$NLP/.cache/huggingface"

VLLM_PIDS=()
LB_PID=""
SHUTTING_DOWN=false

cleanup() {
    SHUTTING_DOWN=true
    echo "Shutting down servers..."

    [ -n "$LB_PID" ] && kill "$LB_PID" 2>/dev/null || true

    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    sleep 5
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "load_balancer.py" 2>/dev/null || true

    for port in $(seq ${BASE_PORT:-9000} $((${BASE_PORT:-9000} + 4))); do
        lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done

    rm -f /tmp/load_balancer_$$.py
    echo "Shutdown complete"
    exit 0
}

trap cleanup EXIT INT TERM

# Check if ports are already in use and kill if needed
MAX_PORT=$((BASE_PORT + 4))
echo "Checking if ports $BASE_PORT-$MAX_PORT are free..."
for port in $(seq $BASE_PORT $MAX_PORT); do
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "Port $port is in use, killing existing process..."
        lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
        sleep 1
    fi
done

# Also kill any stray vllm or load balancer processes
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "load_balancer.py" 2>/dev/null || true
sleep 2

if [ $((N_DEVICES % TP)) -ne 0 ]; then
    echo "Error: n_devices ($N_DEVICES) must be divisible by tensor_parallel ($TP)"
    exit 1
fi

NUM_INSTANCES=$((N_DEVICES / TP))

VLLM_ARGS=(
    --dtype auto
    --max-model-len 12800
    --tensor-parallel-size $TP
    --enable-prefix-caching
    --max-num-seqs 16
    --max-num-batched-tokens 65536
    --enable-chunked-prefill
    --gpu-memory-utilization 0.85
    --kv-cache-dtype auto
    --max-parallel-loading-workers 2
)

[ -n "$MODEL_NAME" ] && VLLM_ARGS+=(--served-model-name "$MODEL_NAME")

echo "Starting $NUM_INSTANCES vLLM instance(s)"
echo "Model: $MODEL_PATH"
echo "TP: $TP | Devices: $N_DEVICES"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + 1 + i))
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
    PORT=$((BASE_PORT + 1 + i))
    while ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; do
        [ "$SHUTTING_DOWN" = true ] && exit 0
        sleep 2
    done
    echo "Server on port $PORT ready"
done

# Create Python load balancer
LB_SCRIPT="/tmp/load_balancer_$$.py"
cat > "$LB_SCRIPT" << 'EOFPY'
#!/usr/bin/env python3
import http.server
import socketserver
import urllib.request
import sys
from itertools import cycle

class LoadBalancingHandler(http.server.BaseHTTPRequestHandler):
    backends = []
    backend_cycle = None
    
    def do_GET(self):
        self._proxy_request()
    
    def do_POST(self):
        self._proxy_request()
    
    def do_PUT(self):
        self._proxy_request()
    
    def do_DELETE(self):
        self._proxy_request()
    
    def _proxy_request(self):
        backend = next(self.backend_cycle)
        
        # Read request body if present
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        
        # Forward request
        url = f"{backend}{self.path}"
        headers = {k: v for k, v in self.headers.items() 
                  if k.lower() not in ['host', 'connection']}
        
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
            with urllib.request.urlopen(req, timeout=600) as response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in ['connection', 'transfer-encoding']:
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.read())
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
    
    def log_message(self, format, *args):
        sys.stdout.write(f"{self.address_string()} - {format % args}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 load_balancer.py <lb_port> <backend_port1> <backend_port2> ...")
        sys.exit(1)

    lb_port = int(sys.argv[1])
    backend_ports = sys.argv[2:]
    LoadBalancingHandler.backends = [f"http://localhost:{port}" for port in backend_ports]
    LoadBalancingHandler.backend_cycle = cycle(LoadBalancingHandler.backends)

    with socketserver.ThreadingTCPServer(("", lb_port), LoadBalancingHandler) as httpd:
        print(f"Load balancer running on port {lb_port}")
        print(f"Backends: {', '.join(LoadBalancingHandler.backends)}")
        httpd.serve_forever()
EOFPY

chmod +x "$LB_SCRIPT"

# Build port list for load balancer
LB_PORTS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    LB_PORTS+=($((BASE_PORT + 1 + i)))
done

echo "Starting Python load balancer..."
python3 "$LB_SCRIPT" "$BASE_PORT" "${LB_PORTS[@]}" &
LB_PID=$!

sleep 2

echo ""
echo "============================================"
echo "All servers started successfully!"
echo "============================================"
echo "Load balancer: http://localhost:$BASE_PORT"
echo "Backend instances:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    echo "  Instance $((i + 1)): http://localhost:$((BASE_PORT + 1 + i))"
done
echo ""
echo "Press Ctrl+C to stop all servers"
echo "============================================"

wait