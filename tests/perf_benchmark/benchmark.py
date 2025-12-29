import argparse
import subprocess
import sys
import os
import time
import signal
import psutil
import threading
import json
import re
from datetime import datetime

# Configuration
REPO_ROOT = "/Users/zhangshulai/Desktop/code_repo"
VLLM_PATH = os.path.join(REPO_ROOT, "vllm")
SWE_BENCH_PATH = os.path.join(REPO_ROOT, "SWE-bench")

def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VLLM_PATH}:{SWE_BENCH_PATH}:{env.get('PYTHONPATH', '')}"
    return env

class SystemMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.cpu_usage = []
        self.cache_metrics = []
        self.monitor_thread = None
        self.perf_process = None

    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.start()
        
        # Start perf stat for cache metrics
        # Note: perf might require sudo or specific permissions. 
        # We try to run it, if it fails, we log a warning.
        try:
            cmd = ["perf", "stat", "-I", str(int(self.interval * 1000)), "-e", "cache-references,cache-misses", "-a"]
            self.perf_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.perf_thread = threading.Thread(target=self._monitor_perf)
            self.perf_thread.start()
        except FileNotFoundError:
            print("Warning: 'perf' command not found. Cache metrics will not be collected.")

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.perf_process:
            self.perf_process.terminate()
            if self.perf_thread:
                self.perf_thread.join()

    def _monitor_cpu(self):
        while self.running:
            self.cpu_usage.append({
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None)
            })
            time.sleep(self.interval)

    def _monitor_perf(self):
        if not self.perf_process:
            return
        
        while self.running and self.perf_process.poll() is None:
            line = self.perf_process.stdout.readline()
            if not line:
                break
            # Parse perf output
            # Example:     1.001234567      1,234,567      cache-references
            parts = line.split()
            if len(parts) >= 3:
                try:
                    ts = float(parts[0])
                    val = parts[1].replace(',', '')
                    metric = parts[2]
                    self.cache_metrics.append({
                        "timestamp": ts,
                        "metric": metric,
                        "value": val
                    })
                except ValueError:
                    pass

    def get_results(self):
        return {
            "cpu_usage": self.cpu_usage,
            "cache_metrics": self.cache_metrics
        }

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
    # Redirect stdout/stderr to avoid cluttering console, or keep it for debug
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

def wait_for_vllm(port, timeout=600):
    print(f"Waiting for vLLM server on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # We can use curl or python requests
            # Using subprocess curl to avoid dependency if requests not installed (though it likely is)
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                print("vLLM server is ready.")
                return True
        except Exception as e:
            pass
        
        if time.time() - start_time > 10 and (time.time() - start_time) % 30 == 0:
             print(f"Still waiting... ({int(time.time() - start_time)}s)")
        time.sleep(5)
    
    print("Timeout waiting for vLLM server.")
    return False

def start_swe_bench(env):
    print("Starting SWE-bench evaluation (background load)...")
    # Using a command that generates CPU load. 
    # We use 'gold' predictions path to just run validation which is CPU intensive.
    # We need to make sure we have a valid instance id.
    # If we don't have the dataset downloaded, this might fail.
    # For the purpose of this script, we assume the environment is set up.
    # If not, we might want to run a dummy CPU stress test instead?
    # The user specifically asked for "SWE-bench evaluation".
    
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", "gold",
        "--max_workers", "16", # High concurrency to stress CPU
        "--instance_ids", "sympy__sympy-20590", # Example instance
        "--run_id", "perf_test_colocated",
        "--namespace", "swebench" 
    ]
    
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process

def run_benchmark_client(model_name, port, env):
    print("Running vLLM benchmark client...")
    bench_script = os.path.join(VLLM_PATH, "benchmarks/benchmark_serving.py")
    
    # Using random dataset for simplicity and reproducibility without external files
    cmd = [
        sys.executable, bench_script,
        "--backend", "openai",
        "--model", model_name,
        "--port", str(port),
        "--dataset-name", "random",
        "--num-prompts", "500",
        "--random-input-len", "1024",
        "--random-output-len", "512",
        "--request-rate", "8" # Adjust based on expected throughput
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result.stdout

def parse_metrics(output):
    # Parse stdout from benchmark_serving.py
    metrics = {}
    # Example output patterns (need to verify against actual output)
    # "Mean TTFT: 123.45 ms"
    # "Mean TPOT: 12.34 ms"
    # "Request throughput: 5.67 req/s"
    
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
    parser = argparse.ArgumentParser(description="vLLM Performance Benchmark with Optional Co-location")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Model name or path")
    parser.add_argument("--co-located", action="store_true", help="Run with SWE-bench co-location")
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
            # Print stderr
            print(vllm_process.stderr.read())
            return

        # 2. Start System Monitor
        monitor = SystemMonitor()
        monitor.start()

        swe_process = None
        if args.co_located:
            # 3. Start SWE-bench (Optional)
            swe_process = start_swe_bench(env)
            # Give it a moment to ramp up
            time.sleep(10)
            if swe_process.poll() is not None:
                print("Warning: SWE-bench process exited early. It might not be running correctly.")

        # 4. Run Benchmark
        client_output = run_benchmark_client(args.model, args.port, env)
        print("Benchmark finished.")
        print(client_output)

        # 5. Stop everything
        monitor.stop()
        
        if swe_process:
            swe_process.terminate()
            swe_process.wait()

        # 6. Save Results
        metrics = parse_metrics(client_output)
        system_data = monitor.get_results()
        
        result_data = {
            "config": vars(args),
            "metrics": metrics,
            "system_data": system_data,
            "raw_output": client_output
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "colocated" if args.co_located else "standalone"
        filename = os.path.join(args.output_dir, f"benchmark_{mode}_{timestamp}.json")
        
        with open(filename, "w") as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Results saved to {filename}")

    finally:
        # Ensure vLLM is killed
        vllm_process.terminate()
        vllm_process.wait()
        # Also kill any lingering processes if needed

if __name__ == "__main__":
    main()
