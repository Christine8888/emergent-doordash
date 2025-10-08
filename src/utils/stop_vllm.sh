#!/bin/bash

echo "Stopping vLLM servers and load balancer..."

# Kill load balancer
pkill -f "load_balancer.py" 2>/dev/null || true

# Kill vLLM servers
pkill -f "vllm serve" 2>/dev/null || true

# Wait a bit for graceful shutdown
sleep 3

# Force kill any remaining processes on ports 9000-9004
for port in {9000..9004}; do
    lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done

# Clean up temp files
rm -f /tmp/load_balancer_*.py

echo "vLLM servers stopped"