import argparse
import subprocess
import sys
import os
import json
from datetime import datetime

# Ensure we can import from utils in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import SystemMonitor, get_env

def run_swe_bench(env, instance_id="sympy__sympy-20590", max_workers=16):
    print(f"Starting SWE-bench evaluation for {instance_id} with {max_workers} workers...")
    
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", "gold",
        "--max_workers", str(max_workers),
        "--instance_ids", instance_id,
        "--run_id", "perf_test_swe",
        "--namespace", "swebench" 
    ]
    
    # Run and stream output to console
    process = subprocess.run(cmd, env=env)
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="SWE-bench Performance Monitor")
    parser.add_argument("--instance-id", type=str, default="sympy__sympy-20590", help="SWE-bench instance ID")
    parser.add_argument("--max-workers", type=int, default=16, help="Max workers for evaluation")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = get_env()
    
    # 1. Start System Monitor
    monitor = SystemMonitor()
    monitor.start()

    # 2. Run SWE-bench
    try:
        return_code = run_swe_bench(env, args.instance_id, args.max_workers)
        print(f"SWE-bench finished with return code {return_code}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # 3. Stop Monitor
        monitor.stop()

    # 4. Save Results
    system_data = monitor.get_results()
    
    result_data = {
        "config": vars(args),
        "system_data": system_data
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_dir, f"swe_bench_metrics_{timestamp}.json")
    
    with open(filename, "w") as f:
        json.dump(result_data, f, indent=2)
        
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
