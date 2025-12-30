import os
import time
import threading
import subprocess
import psutil

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
        self.perf_thread = None

    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.start()
        
        # Start perf stat for cache metrics
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
