# vLLM Performance Benchmark on Ascend

This module tests the performance of a vLLM server and SWE-bench evaluation separately or co-located (by running both scripts).

## Prerequisites

- Ascend 910B node with Kunpeng 920 CPU
- OpenEuler OS
- Python environment with `vllm` and `swebench` installed
- `perf` tool installed (for cache metrics)

## Usage

### 1. Run vLLM Benchmark

This script starts a vLLM server, runs a benchmark client against it, and collects system metrics.

```bash
python run_vllm_benchmark.py --model "Qwen/Qwen2.5-3B"
```

### 2. Run SWE-bench Evaluation

This script runs the SWE-bench evaluation (CPU intensive) and collects system metrics.

```bash
python run_swe_bench.py --instance-id "sympy__sympy-20590"
```

### Co-located Testing

To test performance interference, run both scripts simultaneously in separate terminals.

**Terminal 1 (Background Load):**
```bash
python run_swe_bench.py
```

**Terminal 2 (Measured Workload):**
```bash
python run_vllm_benchmark.py --model "Qwen/Qwen2.5-3B"
```

## Metrics

Results are saved in the `results` directory as JSON files.

- **vLLM Results**: Contains TTFT, TPOT, Throughput, CPU usage, and Cache metrics.
- **SWE-bench Results**: Contains CPU usage and Cache metrics during the evaluation.
