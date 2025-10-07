from vllm_utils import VLLMServer, start_multi_instance, stop_all_servers

# Example 1: Single server
server = VLLMServer(
    model_path="/workspace/model",
    port=9000,
    tensor_parallel=1,
    gpu_ids=[0],
)
server.start()

# Generate text
response = server.generate("Hello, how are you?", max_tokens=50)
print(response)

# Chat
messages = [{"role": "user", "content": "What is 2+2?"}]
response = server.chat(messages)
print(response)

server.stop()

# Example 2: Multiple servers with load balancing
servers = start_multi_instance(
    model_path="/workspace/model",
    n_devices=4,
    tensor_parallel=2,  # 2 instances, each with TP=2
    base_port=9001,
    model_name="my-model",
)

# Use servers[0], servers[1], etc. or implement load balancing
for server in servers:
    print(f"Server URL: {server.url}")

stop_all_servers(servers)
