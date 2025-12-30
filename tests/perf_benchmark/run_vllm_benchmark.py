import argparse
import subprocess
import sys
import os
import time
import json
import re
from datetime import datetime

# Ensure we can import from utils in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import SystemMonitor, get_env, VLLM_PATH

def start_vllm_server(model_name, port, env):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "4096",
        "--disable-log-requests"
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_vllm(port, timeout=600):
    print(f"Waiting for vLLM server on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                print("vLLM server is ready.")
                return True
        except Exception:
            pass
        
        if time.time() - start_time > 10 and (time.time() - start_time) % 30 == 0:
             print(f"Still waiting... ({int(time.time() - start_time)}s)")
        time.sleep(5)
    
    print("Timeout waiting for vLLM server.")
    return False

def run_benchmark_client(model_name, port, env):
    print("Running vLLM benchmark client...")
    bench_script = os.path.join(VLLM_PATH, "benchmarks/benchmark_serving.py")
    
    cmd = [
        sys.executable, bench_script,
        "--backend", "openai",
        "--model", model_name,
        "--port", str(port),
        "--dataset-name", "random",
        "--num-prompts", "500",
        "--random-input-len", "1024",
        "--random-output-len", "512",
        "--request-rate", "8"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result.stdout

def parse_metrics(output):
    metrics = {}
    ttft_match = re.search(r"Mean TTFT:\s+([\d\.]+)\s+ms", output)
    if ttft_match:
        metrics["mean_ttft_ms"] = float(ttft_match.group(1))
        
    tpot_match = re.search(r"Mean TPOT:\s+([\d\.]+)\s+ms", output)
    if tpot_match:
        metrics["mean_tpot_ms"] = float(tpot_match.group(1))
        
    throughput_match = re.search(r"Request throughput:\s+([\d\.]+)\s+req/s", output)
    if throughput_match:
        metrics["throughput_req_s"] = float(throughput_match.group(1))
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="vLLM Performance Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = get_env()
    
    # 1. Start vLLM
    vllm_process = start_vllm_server(args.model, args.port, env)
    
    try:
        if not wait_for_vllm(args.port):
            print("Failed to start vLLM server. Check logs.")
            print(vllm_process.stderr.read())
            return

        # 2. Start System Monitor
        monitor = SystemMonitor()
        monitor.start()

        # 3. Run Benchmark
        client_output = run_benchmark_client(args.model, args.port, env)
        print("Benchmark finished.")
        print(client_output)

        # 4. Stop Monitor
        monitor.stop()
        
        # 5. Save Results
        metrics = parse_metrics(client_output)
        system_data = monitor.get_results()
        
        result_data = {
            "config": vars(args),
            "metrics": metrics,
            "system_data": system_data,
            "raw_output": client_output
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.output_dir, f"vllm_benchmark_{timestamp}.json")
        
        with open(filename, "w") as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Results saved to {filename}")

    finally:
        vllm_process.terminate()
        vllm_process.wait()

if __name__ == "__main__":
    main()
