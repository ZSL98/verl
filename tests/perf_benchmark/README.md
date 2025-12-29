# vLLM Performance Benchmark on Ascend

This module tests the performance of a vLLM server with or without co-located SWE-bench evaluation.

## Prerequisites

- Ascend 910B node with Kunpeng 920 CPU
- OpenEuler OS
- Python environment with `vllm` and `swebench` installed (or present in the workspace)
- `perf` tool installed (for cache metrics)

## Usage

Run the benchmark script:

```bash
# Standalone vLLM benchmark
python benchmark.py --model "Qwen/Qwen2.5-3B"

# Co-located with SWE-bench
python benchmark.py --model "Qwen/Qwen2.5-3B" --co-located
```

## Metrics

The script collects:
1.  **SLO-level metrics**: TTFT, TPOT, Throughput (via `vllm/benchmarks/benchmark_serving.py`)
2.  **Hardware-level metrics**: CPU utilization trace, CPU cache hit ratio (via `psutil` and `perf`)

Results are saved in the `results` directory as JSON files.

## Configuration

- Modify `benchmark.py` to adjust `vllm` arguments (tensor parallelism, dtype, etc.) or benchmark client parameters (request rate, num prompts).
- Ensure `SWE-bench` is properly set up if running with `--co-located`.
