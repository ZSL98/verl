import argparse
import subprocess
import sys
import os
import time
import json
import re
from datetime import datetime

# Configuration
REPO_ROOT = "/Users/zhangshulai/Desktop/code_repo"
VLLM_PATH = os.path.join(REPO_ROOT, "vllm")

def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VLLM_PATH}:{env.get('PYTHONPATH', '')}"
    return env

class VLLMBenchmark:
    def __init__(self, model_name, port, env):
        self.model_name = model_name
        self.port = port
        self.env = env
        self.process = None

    def start_server(self):
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--trust-remote-code",
            "--tensor-parallel-size", "1",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "4096",
            "--disable-log-requests"
        ]
        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return self.process.pid

    def wait_for_ready(self, timeout=600):
        print(f"Waiting for vLLM server on port {self.port}...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    ["curl", "-s", f"http://localhost:{self.port}/health"],
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
        return False

    def run_client(self):
        print("Running vLLM benchmark client...")
        bench_script = os.path.join(VLLM_PATH, "benchmarks/benchmark_serving.py")
        cmd = [
            sys.executable, bench_script,
            "--backend", "openai",
            "--model", self.model_name,
            "--port", str(self.port),
            "--dataset-name", "random",
            "--num-prompts", "500",
            "--random-input-len", "1024",
            "--random-output-len", "512",
            "--request-rate", "8"
        ]
        result = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        return result.stdout

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

    @staticmethod
    def parse_metrics(output):
        metrics = {}
        ttft_match = re.search(r"Mean TTFT:\s+([\d\.]+)\s+ms", output)
        if ttft_match: metrics["mean_ttft_ms"] = float(ttft_match.group(1))
        tpot_match = re.search(r"Mean TPOT:\s+([\d\.]+)\s+ms", output)
        if tpot_match: metrics["mean_tpot_ms"] = float(tpot_match.group(1))
        throughput_match = re.search(r"Request throughput:\s+([\d\.]+)\s+req/s", output)
        if throughput_match: metrics["throughput_req_s"] = float(throughput_match.group(1))
        return metrics

def main():
    parser = argparse.ArgumentParser(description="vLLM Container Runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = get_env()
    
    vllm_bench = VLLMBenchmark(args.model, args.port, env)
    
    try:
        # 1. Start vLLM
        vllm_bench.start_server()
        if not vllm_bench.wait_for_ready():
            print("vLLM failed to start")
            return

        # 2. Warmup
        print("Warming up for 10 seconds...")
        time.sleep(10)

        # 3. Run Benchmark
        output = vllm_bench.run_client()
        metrics = VLLMBenchmark.parse_metrics(output)
        
        # 4. Save Results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(args.output_dir, f"vllm_metrics_{timestamp}.json")
        
        result_data = {
            "config": vars(args),
            "metrics": metrics,
            "raw_output": output
        }
        
        with open(filename, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"vLLM metrics saved to {filename}")

    finally:
        vllm_bench.stop()

if __name__ == "__main__":
    main()
