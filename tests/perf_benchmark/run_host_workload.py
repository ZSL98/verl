import argparse
import subprocess
import sys
import os
import time
import psutil
import threading
import csv
from datetime import datetime

# Configuration
REPO_ROOT = "/Users/zhangshulai/Desktop/code_repo"
SWE_BENCH_PATH = os.path.join(REPO_ROOT, "SWE-bench")

def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SWE_BENCH_PATH}:{env.get('PYTHONPATH', '')}"
    return env

class SystemMonitor:
    def __init__(self, interval=1.0, target_pids=None):
        self.interval = interval
        self.target_pids = target_pids or {} # {'name': pid}
        self.running = False
        self.cpu_usage = []
        self.process_stats = {name: [] for name in self.target_pids}
        self.cache_metrics = []
        self.monitor_thread = None
        self.perf_process = None

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
            timestamp = time.time()
            # Global CPU
            self.cpu_usage.append({
                "timestamp": timestamp,
                "cpu_percent": psutil.cpu_percent(interval=None)
            })
            
            # Per-process CPU
            for name, pid in self.target_pids.items():
                try:
                    proc = psutil.Process(pid)
                    cpu = proc.cpu_percent(interval=None)
                    mem = proc.memory_info().rss / (1024 * 1024) # MB
                    
                    # Sum up children
                    children = proc.children(recursive=True)
                    for child in children:
                        try:
                            cpu += child.cpu_percent(interval=None)
                            mem += child.memory_info().rss / (1024 * 1024)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    self.process_stats[name].append({
                        "timestamp": timestamp,
                        "cpu_percent": cpu,
                        "memory_mb": mem
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            time.sleep(self.interval)

    def _monitor_perf(self):
        if not self.perf_process:
            return
        
        while self.running and self.perf_process.poll() is None:
            line = self.perf_process.stdout.readline()
            if not line:
                break
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
            "process_stats": self.process_stats,
            "cache_metrics": self.cache_metrics
        }

class SWEBenchBenchmark:
    def __init__(self, env):
        self.env = env
        self.process = None

    def start(self):
        print("Starting SWE-bench evaluation...")
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--predictions_path", "gold",
            "--max_workers", "16",
            "--instance_ids", "sympy__sympy-20590",
            "--run_id", "perf_test",
            "--namespace", "swebench" 
        ]
        self.process = subprocess.Popen(cmd, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self.process.pid

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

def save_csv(results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hw_filename = os.path.join(output_dir, f"hardware_metrics_{timestamp}.csv")
    
    system_data = results.get("system_data", {})
    cpu_usage = system_data.get("cpu_usage", [])
    process_stats = system_data.get("process_stats", {})
    cache_metrics = system_data.get("cache_metrics", [])
    
    with open(hw_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["timestamp", "system_cpu_percent"]
        proc_names = list(process_stats.keys())
        for name in proc_names:
            header.append(f"{name}_cpu_percent")
            header.append(f"{name}_memory_mb")
        writer.writerow(header)
        
        for i, cpu_point in enumerate(cpu_usage):
            ts = cpu_point["timestamp"]
            row = [ts, cpu_point["cpu_percent"]]
            for name in proc_names:
                stats = process_stats.get(name, [])
                if i < len(stats):
                    row.append(stats[i]["cpu_percent"])
                    row.append(stats[i]["memory_mb"])
                else:
                    row.append("")
                    row.append("")
            writer.writerow(row)
            
    print(f"Hardware metrics saved to {hw_filename}")
    
    if cache_metrics:
        cache_filename = os.path.join(output_dir, f"cache_metrics_{timestamp}.csv")
        with open(cache_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "metric", "value"])
            for item in cache_metrics:
                writer.writerow([item["timestamp"], item["metric"], item["value"]])
        print(f"Cache metrics saved to {cache_filename}")

def main():
    parser = argparse.ArgumentParser(description="SWE-bench Host Runner & Monitor")
    parser.add_argument("--duration", type=int, default=120, help="Duration to run monitoring (seconds)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = get_env()
    
    swe_bench = SWEBenchBenchmark(env)
    monitor = None
    
    try:
        # 1. Start SWE-bench
        pid = swe_bench.start()
        
        # 2. Start Monitoring
        print(f"SWE-bench started (PID: {pid}). Starting system monitor...")
        print(f"Monitoring for {args.duration} seconds. Run the vLLM workload in the container NOW.")
        
        monitor = SystemMonitor(target_pids={"swebench": pid})
        monitor.start()
        
        # 3. Wait for duration
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopping early...")

        # 4. Stop and Save
        monitor.stop()
        results = {"system_data": monitor.get_results()}
        save_csv(results, args.output_dir)

    finally:
        if monitor: monitor.stop()
        if swe_bench: swe_bench.stop()

if __name__ == "__main__":
    main()
