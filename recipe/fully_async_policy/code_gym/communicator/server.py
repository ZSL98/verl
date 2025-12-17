from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.responses import JSONResponse
import subprocess
import traceback
import shlex
import time
import re
import random
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化FastAPI应用
app = FastAPI(title="NUMA Bind Task Executor", version="1.0")

# ======================== 核心配置 ========================
# 1. API鉴权Key（调用方需携带）
AUTH_API_KEY = "container-a-secure-key-2025"

# 2. 安全配置：允许的基础命令（绑核/采样相关）
ALLOWED_BASE_COMMANDS = {
    "numactl", "ps", "lscpu", "perf", "taskset", "kill", "grep", "top"
}

# 3. 任务队列（FIFO）+ 锁（保证线程安全）
task_queue: List[Dict[str, Any]] = []
queue_lock = threading.Lock()
is_processing = False
completed_results: Dict[str, Any] = {}
processing_requests = set()
process_registry_lock = threading.Lock()
tracked_processes: Dict[int, "TrackedProcess"] = {}

# 4. NUMA/CPU合法性校验正则
NUMA_NODE_PATTERN = re.compile(r"^\d+$")  # 数字格式的NUMA节点
CPU_LIST_PATTERN = re.compile(r"^\d+(,\d+)*(-\d+)*$")  # 支持1,2,3 或 0-7格式
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
BENCHMARK_BIN_DIR = BENCHMARK_DIR / "cpubench"
DISK_TEST_FILE = BENCHMARK_DIR / "disk_test.tmp"
TEST_DURATION = 1200
LOAD_COUNT_RANGE: Dict[str, tuple] = {
    "compute": (1, 5),
    "mem": (1, 8),
    "cache": (0, 5),
    "disk": (0, 5)
}
COMPUTE_THREADS = 48
MEM_THREADS = 48
MEM_SIZE_MB = 4096
MEM_GRANULARITY = 64
MEM_SEQUENTIAL = 0
CACHE_THREADS = 48
CACHE_SIZE_MB = 40
DISK_THREADS = 48
DISK_FILE_SIZE_MB = 1024
DISK_BLOCK_SIZE_KB = 8
DISK_SEQUENTIAL = 0
DISK_READ_ONLY = 0

# benchmark 进程采样临时文件目录（由 load_common.h 写入）
CODEGYM_SAMPLE_DIR = Path(os.environ.get("CODEGYM_SAMPLE_DIR", "/tmp/codegym_samples"))
CODEGYM_SAMPLE_PREFIX = "sample_"
CODEGYM_SAMPLE_SUFFIX = ".log"

# ======================== 数据结构定义 ========================
@dataclass
class BindCommandResult:
    """单条绑核指令的执行与采样结果"""
    command: str
    pid: Optional[int]
    bind_success: bool
    sample_results: Dict[str, Any]
    exit_code: int
    reward: Optional[Dict[str, Any]] = None
    error_msg: str = ""

@dataclass
class BindTaskResult:
    """绑核任务（包含多条指令）的整体结果"""
    request_id: str
    success: bool
    command_results: List[BindCommandResult]
    reward: Optional[Dict[str, Any]] = None
    error_msg: str = ""


@dataclass
class TrackedProcess:
    """记录由当前server启动的进程，便于统一停止"""
    proc: subprocess.Popen
    command: str
    source: str
    start_time: float

# ======================== 工具函数 ========================
def validate_numa_cpu(numa_node: str, cpu_list: str) -> bool:
    """校验NUMA节点和CPU核心是否合法（基于lscpu输出）"""
    try:
        # 校验格式
        if not NUMA_NODE_PATTERN.match(numa_node):
            return False
        if not CPU_LIST_PATTERN.match(cpu_list):
            return False
        
        # 校验实际存在的NUMA节点
        lscpu_result = subprocess.run(
            ["lscpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        numa_nodes = re.findall(r"NUMA node\(s\):\s+(\d+)", lscpu_result.stdout)
        if not numa_nodes or int(numa_node) >= int(numa_nodes[0]):
            return False
        
        # 校验CPU核心范围（简化版：仅检查最大CPU数）
        cpu_max = re.findall(r"CPU\(s\):\s+(\d+)", lscpu_result.stdout)
        if not cpu_max:
            return False
        max_cpu = int(cpu_max[0]) - 1  # CPU编号从0开始
        # 解析CPU列表中的所有核心
        cpu_parts = cpu_list.replace(",", "-").split("-")
        for cpu in cpu_parts:
            if cpu and int(cpu) > max_cpu:
                return False
        
        return True
    except Exception:
        return False

def execute_shell_command(cmd_parts: List[str], timeout: int = 10) -> Dict[str, str]:
    """
    执行单个shell命令（安全模式，无shell注入）
    :param cmd_parts: 命令拆分列表（如["ps", "-ef"]）
    :param timeout: 超时时间
    :return: 包含stdout/stderr/exit_code的字典
    """
    try:
        # 校验基础命令是否在白名单
        if cmd_parts[0] not in ALLOWED_BASE_COMMANDS:
            return {
                "exit_code": -3,
                "stdout": "",
                "stderr": f"禁止执行命令：{cmd_parts[0]}（仅允许{ALLOWED_BASE_COMMANDS}）"
            }
        
        result = subprocess.run(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            encoding="utf-8",
            errors="ignore"
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"命令执行超时（{timeout}秒）"
        }
    except Exception as e:
        return {
            "exit_code": -2,
            "stdout": "",
            "stderr": f"命令执行失败：{str(e)}"
        }

def _sample_file_for_pid(pid: int) -> Path:
    return CODEGYM_SAMPLE_DIR / f"{CODEGYM_SAMPLE_PREFIX}{pid}{CODEGYM_SAMPLE_SUFFIX}"


def get_tracked_pids() -> List[int]:
    cleanup_finished_processes()
    with process_registry_lock:
        return list(tracked_processes.keys())

def get_workload_processes() -> List[Dict[str, Any]]:
    """返回由 server 启动的 benchmark 负载进程信息（PID/命令/来源）。"""
    cleanup_finished_processes()
    with process_registry_lock:
        items = [
            (pid, tracked)
            for pid, tracked in tracked_processes.items()
            if tracked.source.startswith("benchmark:")
        ]
    workloads: List[Dict[str, Any]] = []
    for pid, tracked in sorted(items, key=lambda it: it[0]):
        workloads.append(
            {
                "pid": pid,
                "command": tracked.command,
                "source": tracked.source,
                "start_time": tracked.start_time,
            }
        )
    return workloads


def _parse_perf_stat_csv(stderr_text: str) -> Dict[str, Optional[float]]:
    """解析 perf stat -x, 输出，返回 event_name -> value（无法解析则为 None / 缺失）"""
    counters: Dict[str, Optional[float]] = {}
    for line in stderr_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        value_str, _, event_name = parts[0], parts[1], parts[2]
        if not event_name or event_name == "time elapsed":
            continue
        if value_str.startswith("<") and value_str.endswith(">"):
            counters[event_name] = None
            continue
        normalized = value_str.replace(",", "").strip()
        try:
            counters[event_name] = float(normalized)
        except ValueError:
            continue
    return counters


def _get_perf_counter_value(counters: Dict[str, Optional[float]], event_name: str) -> Optional[float]:
    """兼容 perf 输出 event 名带 :u/:k 等修饰符的情况。"""
    if event_name in counters:
        return counters[event_name]
    for key, value in counters.items():
        base = key.split(":", 1)[0]
        if base == event_name:
            return value
    return None


def _perf_output_indicates_unsupported(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    return (
        "not supported" in lowered
        or "unknown event" in lowered
        or "failed to find event" in lowered
        or "no such file or directory" in lowered
    )


def _perf_output_indicates_permission_issue(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    return ("permission" in lowered and "denied" in lowered) or "no permission" in lowered


def _perf_sample_l3_hit_rate_for_pid(
    pid: int,
    sample_seconds: float,
    loads_event: str,
    misses_event: str,
) -> Dict[str, Any]:
    cmd = [
        "perf",
        "stat",
        "-x",
        ",",
        "-e",
        f"{loads_event},{misses_event}",
        "-p",
        str(pid),
        "--",
        "sleep",
        str(sample_seconds),
    ]
    raw = execute_shell_command(cmd, timeout=max(6, int(sample_seconds) + 5))
    perf_text = f"{raw.get('stderr', '')}\n{raw.get('stdout', '')}"
    counters = _parse_perf_stat_csv(perf_text)
    loads = _get_perf_counter_value(counters, loads_event)
    misses = _get_perf_counter_value(counters, misses_event)
    hit_rate: Optional[float] = None
    if loads is not None and misses is not None and loads > 0:
        hit_rate = max(0.0, min(1.0, (loads - misses) / loads))
    return {
        "exit_code": raw.get("exit_code", -2),
        "loads_event": loads_event,
        "misses_event": misses_event,
        "loads": loads,
        "misses": misses,
        "hit_rate": hit_rate,
        "stderr": raw.get("stderr", ""),
        "stdout": raw.get("stdout", ""),
    }


def _perf_sample_l3_hit_rate_for_pid_with_retry(
    pid: int,
    sample_seconds: float,
    loads_event: str,
    misses_event: str,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    perf stat 偶发会输出 <not counted> 导致 loads/misses 解析为 None。
    这里通过延长采样窗口重试，尽量保证返回可用于计算 hit_rate 的计数结果。
    """
    last: Optional[Dict[str, Any]] = None
    for attempt in range(1, max_attempts + 1):
        duration = sample_seconds * (2 ** (attempt - 1))
        sample = _perf_sample_l3_hit_rate_for_pid(pid, duration, loads_event, misses_event)
        sample["attempt"] = attempt
        sample["sample_seconds"] = duration
        last = sample

        loads = sample.get("loads")
        misses = sample.get("misses")
        if loads is not None and misses is not None and loads > 0:
            return sample

        stderr_text = str(sample.get("stderr", "") or "")
        if _perf_output_indicates_permission_issue(stderr_text) or _perf_output_indicates_unsupported(stderr_text):
            break

    return last or {
        "exit_code": -2,
        "loads_event": loads_event,
        "misses_event": misses_event,
        "loads": None,
        "misses": None,
        "hit_rate": None,
        "stderr": "perf 采样失败：无可用结果",
        "stdout": "",
        "attempt": 0,
        "sample_seconds": sample_seconds,
    }


def sample_workload_l3_hit_rate(
    pids: List[int],
    sample_seconds: float = 0.5,
    max_workers: int = 6,
) -> Dict[str, Any]:
    """对每个 PID 使用 perf 采样 L3 命中率（hit_rate = 1 - misses / loads）。"""
    if not pids:
        return {
            "exit_code": 0,
            "results": {},
            "loads_event": "LLC-loads",
            "misses_event": "LLC-load-misses",
            "stderr": "",
        }

    event_candidates = [
        ("LLC-loads", "LLC-load-misses"),
        ("cache-references", "cache-misses"),
    ]
    loads_event, misses_event = event_candidates[0]

    # 先用首个PID探测一次事件是否可用；仅当明确“不支持”时回退到更通用的 cache-* 事件
    probe = _perf_sample_l3_hit_rate_for_pid(pids[0], min(sample_seconds, 0.2), loads_event, misses_event)
    if _perf_output_indicates_unsupported(str(probe.get("stderr", "") or "")):
        loads_event, misses_event = event_candidates[1]

    results: Dict[str, Any] = {}
    errors: List[str] = []
    worker_count = max(1, min(max_workers, len(pids)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_perf_sample_l3_hit_rate_for_pid_with_retry, pid, sample_seconds, loads_event, misses_event): pid
            for pid in pids
        }
        for future in as_completed(future_map):
            pid = future_map[future]
            try:
                results[str(pid)] = future.result()
            except Exception as exc:
                errors.append(f"PID {pid}: perf 采样失败：{exc}")
                results[str(pid)] = {
                    "exit_code": -2,
                    "loads_event": loads_event,
                    "misses_event": misses_event,
                    "loads": None,
                    "misses": None,
                    "hit_rate": None,
                    "stderr": str(exc),
                }

    return {
        "exit_code": 0 if not errors else -1,
        "loads_event": loads_event,
        "misses_event": misses_event,
        "results": results,
        "stderr": "\n".join(errors),
    }


def _parse_top_cpu_percent(output_text: str) -> Dict[int, float]:
    cpu_index: Optional[int] = None
    cpu_by_pid: Dict[int, float] = {}

    for line in output_text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue

        if stripped.startswith("PID "):
            cols = stripped.split()
            if "%CPU" in cols:
                cpu_index = cols.index("%CPU")
            continue

        if not stripped[0].isdigit():
            continue

        parts = stripped.split()
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        idx = cpu_index if cpu_index is not None else 8
        if len(parts) <= idx:
            continue
        try:
            cpu_by_pid[pid] = float(parts[idx].replace("%", ""))
        except ValueError:
            continue

    return cpu_by_pid


def sample_workload_cpu_percent_top(
    pids: List[int],
    delay_seconds: float = 0.2,
    iterations: int = 2,
    chunk_size: int = 20,
) -> Dict[str, Any]:
    """使用 top 批量采样每个 PID 的 CPU 利用率（%CPU）。"""
    if not pids:
        return {"exit_code": 0, "cpu_percent": {}, "stderr": "", "stdout": ""}

    cpu_percent: Dict[str, Optional[float]] = {str(pid): None for pid in pids}
    errors: List[str] = []
    raw_outputs: List[str] = []

    for i in range(0, len(pids), chunk_size):
        chunk = pids[i : i + chunk_size]
        cmd = [
            "top",
            "-b",
            "-n",
            str(iterations),
            "-d",
            str(delay_seconds),
            "-p",
            ",".join(str(pid) for pid in chunk),
        ]
        raw = execute_shell_command(cmd, timeout=max(5, int(delay_seconds * iterations) + 3))
        if raw.get("stdout"):
            raw_outputs.append(raw["stdout"])
        if raw.get("exit_code") != 0:
            errors.append(raw.get("stderr", "") or f"top 采样失败：exit_code={raw.get('exit_code')}")
            continue
        parsed = _parse_top_cpu_percent(raw.get("stdout", ""))
        for pid, value in parsed.items():
            cpu_percent[str(pid)] = value

    return {
        "exit_code": 0 if not errors else -1,
        "cpu_percent": cpu_percent,
        "stderr": "\n".join(e for e in errors if e),
        "stdout": "\n\n".join(raw_outputs[-1:]),
    }


def collect_latest_benchmark_samples(pids: Optional[List[int]] = None) -> Dict[str, Any]:
    """从临时文件读取每个进程最近一次benchmark采样日志"""
    if pids is None:
        pids = get_tracked_pids()
    lines: List[str] = []
    errors: List[str] = []
    for pid in pids:
        path = _sample_file_for_pid(pid)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                lines.append(f"PID {pid}: {text}")
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f"PID {pid} 读取失败: {exc}")
    return {
        "exit_code": 0 if not errors else -1,
        "stdout": "\n".join(lines),
        "stderr": "\n".join(errors),
    }

OPS_PER_SECOND_PATTERN = re.compile(
    r"PID\s+(?P<pid>\d+):.*?Ops per second:\s*(?P<ops>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def parse_ops_per_second_from_benchmark_latest(latest_log: Dict[str, Any]) -> Dict[int, float]:
    """解析 collect_latest_benchmark_samples 的 stdout，提取每个 PID 的 Ops per second。"""
    stdout = str(latest_log.get("stdout", "") or "")
    ops_by_pid: Dict[int, float] = {}
    for line in stdout.splitlines():
        m = OPS_PER_SECOND_PATTERN.search(line)
        if not m:
            continue
        try:
            pid = int(m.group("pid"))
            ops = float(m.group("ops"))
        except Exception:
            continue
        ops_by_pid[pid] = ops
    return ops_by_pid


def compute_ops_change_rate_reward(
    before_latest: Dict[str, Any],
    after_latest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    reward 计算：
    - 对每个 PID：change_rate = (after_ops - before_ops) / before_ops
    - reward_score：所有有效 PID 的 change_rate 均值
    """
    before_ops = parse_ops_per_second_from_benchmark_latest(before_latest)
    after_ops = parse_ops_per_second_from_benchmark_latest(after_latest)

    per_pid_change_rate: Dict[str, Optional[float]] = {}
    valid_rates: List[float] = []
    stdout_lines: List[str] = []
    for pid, after in after_ops.items():
        before = before_ops.get(pid)
        if before is None or before <= 0:
            per_pid_change_rate[str(pid)] = None
            stdout_lines.append(f"PID {pid}: before_ops=N/A after_ops={after} change_rate=N/A")
            continue
        rate = (after - before) / before
        per_pid_change_rate[str(pid)] = rate
        valid_rates.append(rate)
        stdout_lines.append(f"PID {pid}: before_ops={before} after_ops={after} change_rate={rate}")

    score = sum(valid_rates) / len(valid_rates) if valid_rates else 0.0
    stderr_parts: List[str] = []
    if not valid_rates:
        stderr_parts.append("未找到可用于计算 reward 的有效 PID（缺少 before/after 或 before_ops<=0）")

    return {
        "exit_code": 0 if valid_rates else -1,
        "score": score,
        "stdout": "\n".join(stdout_lines + [f"mean_change_rate: {score}"]),
        "per_pid_change_rate": per_pid_change_rate,
        "before_ops_per_second": {str(pid): val for pid, val in before_ops.items()},
        "after_ops_per_second": {str(pid): val for pid, val in after_ops.items()},
        "stderr": "\n".join(stderr_parts),
    }


def failure_reward(reason: str) -> Dict[str, Any]:
    return {
        "exit_code": -1,
        "score": -1.0,
        "stdout": "",
        "per_pid_change_rate": {},
        "before_ops_per_second": {},
        "after_ops_per_second": {},
        "stderr": reason,
    }


def collect_baseline_sample() -> Dict[str, Any]:
    """采集当前机器的初始 ps/lscpu + workload(pid) 维度的 perf/top 采样"""
    workload_processes = get_workload_processes()
    workload_pids = [item["pid"] for item in workload_processes]
    samples = {
        "ps_ef": execute_shell_command(["ps", "-ef"], timeout=5),
        "lscpu": execute_shell_command(["lscpu"], timeout=5),
        "workload_processes": workload_processes,
        "workload_l3_hit_rate": sample_workload_l3_hit_rate(workload_pids, sample_seconds=0.5),
        "workload_cpu_percent": sample_workload_cpu_percent_top(workload_pids, delay_seconds=0.2, iterations=2),
    }
    samples["benchmark_latest"] = collect_latest_benchmark_samples(workload_pids)
    return samples

def _is_intensive_command(tokens: List[str]) -> bool:
    """判断命令行中是否包含 *intensive 的二进制"""
    for token in tokens:
        name = Path(token).name
        if name.endswith("intensive"):
            return True
    return False


def _pid_is_intensive(pid: int) -> bool:
    """检查给定PID是否对应 *intensive 结尾的任务"""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        for part in cmdline:
            if not part:
                continue
            if Path(part.decode(errors="ignore")).name.endswith("intensive"):
                return True
    except Exception:
        pass
    try:
        ps = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        for line in ps.stdout.splitlines():
            if Path(line.strip()).name.endswith("intensive"):
                return True
    except Exception:
        pass
    return False


def _format_command(command_args: Any) -> str:
    """将Popen的args转换为易读的字符串形式"""
    if isinstance(command_args, (list, tuple)):
        return shlex.join([str(arg) for arg in command_args])
    return str(command_args)


def unregister_tracked_process(pid: int) -> None:
    """从全局跟踪表中移除指定PID"""
    with process_registry_lock:
        tracked_processes.pop(pid, None)


def register_tracked_process(proc: subprocess.Popen, source: str) -> None:
    """记录由server启动的进程，并在退出后自动清理"""
    cmd_str = _format_command(proc.args)
    with process_registry_lock:
        tracked_processes[proc.pid] = TrackedProcess(
            proc=proc,
            command=cmd_str,
            source=source,
            start_time=time.time()
        )

    def _auto_cleanup() -> None:
        try:
            proc.wait()
        except Exception:
            # 退出异常不影响清理
            pass
        unregister_tracked_process(proc.pid)

    threading.Thread(target=_auto_cleanup, daemon=True).start()


def cleanup_finished_processes() -> None:
    """移除已退出的进程，防止跟踪表泄漏"""
    with process_registry_lock:
        finished = [pid for pid, tracked in tracked_processes.items() if tracked.proc.poll() is not None]
        for pid in finished:
            tracked_processes.pop(pid, None)


def stop_all_tracked_processes(timeout: float = 5.0) -> Dict[str, Any]:
    """
    终止当前由server启动并仍在运行的所有进程。
    :return: 汇总信息（停止数量/已退出/错误/详细列表）
    """
    cleanup_finished_processes()
    with process_registry_lock:
        tracked_items = list(tracked_processes.items())

    results: List[Dict[str, Any]] = []
    for pid, tracked in tracked_items:
        proc = tracked.proc
        status: Dict[str, Any] = {
            "pid": pid,
            "command": tracked.command,
            "source": tracked.source
        }
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                    status["status"] = "terminated"
                except subprocess.TimeoutExpired:
                    proc.kill()
                    status["status"] = "killed"
                status["exit_code"] = proc.returncode
            else:
                status["status"] = "already_exited"
                status["exit_code"] = proc.returncode
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        finally:
            unregister_tracked_process(pid)
        results.append(status)

    summary = {
        "total_processed": len(results),
        "stopped": len([r for r in results if r["status"] in ("terminated", "killed")]),
        "already_exited": len([r for r in results if r["status"] == "already_exited"]),
        "errors": len([r for r in results if r["status"] == "error"]),
        "details": results
    }
    with process_registry_lock:
        summary["remaining_after_stop"] = len(tracked_processes)
    return summary


def random_generate_load_counts() -> Dict[str, int]:
    load_counts: Dict[str, int] = {}
    for load_name, (min_cnt, max_cnt) in LOAD_COUNT_RANGE.items():
        load_counts[load_name] = random.randint(min_cnt, max_cnt)
    total = sum(load_counts.values())
    if total == 0:
        print("⚠️ 所有负载随机数量均为0，强制为compute类型分配1个实例")
        load_counts["compute"] = 1
    return load_counts

def check_benchmark_dependencies() -> bool:
    """检查benchmark可执行文件是否存在且可执行"""
    required = [
        BENCHMARK_BIN_DIR / "compute_intensive",
        BENCHMARK_BIN_DIR / "mem_intensive",
        BENCHMARK_BIN_DIR / "cache_sensitive",
        BENCHMARK_BIN_DIR / "io_disk_intensive"
    ]
    missing = [str(p) for p in required if not p.exists() or not os.access(p, os.X_OK)]
    if missing:
        print(f"错误：缺少可执行文件或权限不足：{', '.join(missing)}")
        return False
    return True


def build_load_command(load_name: str) -> List[str]:
    if load_name == "compute":
        return [
            str(BENCHMARK_BIN_DIR / "compute_intensive"),
            "-t", str(COMPUTE_THREADS),
            "-T", str(COMPUTE_THREADS),
            "-f", "0",
            "-d", "0",
            "-r", str(TEST_DURATION)
        ]
    if load_name == "mem":
        return [
            str(BENCHMARK_BIN_DIR / "mem_intensive"),
            "-t", str(MEM_THREADS),
            "-T", str(MEM_THREADS),
            "-M", str(MEM_SIZE_MB),
            "-g", str(MEM_GRANULARITY),
            "-s", str(MEM_SEQUENTIAL),
            "-f", "0",
            "-d", "0",
            "-r", str(TEST_DURATION)
        ]
    if load_name == "cache":
        return [
            str(BENCHMARK_BIN_DIR / "cache_sensitive"),
            "-t", str(CACHE_THREADS),
            "-T", str(CACHE_THREADS),
            "-C", str(CACHE_SIZE_MB),
            "-f", "0",
            "-d", "0",
            "-r", str(TEST_DURATION)
        ]
    if load_name == "disk":
        return [
            str(BENCHMARK_BIN_DIR / "io_disk_intensive"),
            "-t", str(DISK_THREADS),
            "-T", str(DISK_THREADS),
            "-p", str(DISK_TEST_FILE),
            "-F", str(DISK_FILE_SIZE_MB),
            "-b", str(DISK_BLOCK_SIZE_KB),
            "-s", str(DISK_SEQUENTIAL),
            "-R", str(DISK_READ_ONLY),
            "-f", "0",
            "-d", "0",
            "-r", str(TEST_DURATION)
        ]
    raise ValueError(f"未知负载类型：{load_name}")


def start_load_instances(load_counts: Dict[str, int]) -> List[tuple]:
    processes = []
    total_instances = sum(load_counts.values())
    print(f"\n🚀 开始启动 {total_instances} 个benchmark任务进程（按类型随机分配）：")
    for load_name, count in load_counts.items():
        if count <= 0:
            print(f"  - {load_name}: 0 个实例（跳过）")
            continue
        print(f"  - {load_name}: {count} 个实例")
        for idx in range(1, count + 1):
            try:
                cmd = build_load_command(load_name)
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    cwd=str(BENCHMARK_DIR)
                )
                register_tracked_process(proc, source=f"benchmark:{load_name}[{idx}]")
                processes.append((load_name, idx, proc))
                print(f"    ✅ 已启动 {load_name}[{idx}] (PID: {proc.pid})")
            except Exception as e:
                print(f"    ❌ 启动 {load_name}[{idx}] 失败：{str(e)}")
    return processes


def wait_for_processes(processes: List[tuple]) -> None:
    if not processes:
        return
    print(f"\n⌛ 等待所有benchmark实例运行 {TEST_DURATION} 秒...")
    start_time = time.time()
    for load_name, idx, proc in processes:
        try:
            proc.wait(timeout=TEST_DURATION + 5)
            exit_code = proc.returncode
            if exit_code == 0:
                print(f"✅ {load_name}[{idx}] 运行完成 (退出码: {exit_code})")
            else:
                print(f"⚠️ {load_name}[{idx}] 异常退出 (退出码: {exit_code})")
        except subprocess.TimeoutExpired:
            print(f"⚠️ {load_name}[{idx}] 运行超时，强制终止")
            proc.kill()
        finally:
            unregister_tracked_process(proc.pid)
    if DISK_TEST_FILE.exists():
        try:
            os.remove(DISK_TEST_FILE)
            print(f"\n🗑️ 已清理磁盘测试文件：{DISK_TEST_FILE}")
        except Exception as e:
            print(f"⚠️ 清理磁盘测试文件失败：{str(e)}")
    elapsed = time.time() - start_time
    print(f"\n📊 所有benchmark实例运行完成，总耗时：{elapsed:.2f} 秒")


def launch_random_workload() -> tuple[List[tuple], Dict[str, int]]:
    """生成随机负载并启动进程，返回进程列表和数量配置"""
    if not check_benchmark_dependencies():
        return [], {}
    load_counts = random_generate_load_counts()
    print("\n📋 随机生成的benchmark实例数量：")
    for load_name, count in load_counts.items():
        print(f"  - {load_name}: {count} 个")
    processes = start_load_instances(load_counts)
    return processes, load_counts


def start_benchmark_workload() -> None:
    """服务启动时直接在当前进程启动随机benchmark负载"""
    try:
        processes, _ = launch_random_workload()
        if not processes:
            return
        wait_for_processes(processes)
        print("\n🎉 随机多进程benchmark负载完成！")
    except Exception as e:
        print(f"❌ benchmark 任务异常：{str(e)}")


def trigger_random_workload_async(source: str = "manual") -> List[int]:
    """以后台线程方式启动一次随机benchmark负载，返回新进程的PID列表"""
    try:
        processes, _ = launch_random_workload()
        if not processes:
            return []
        threading.Thread(
            target=wait_for_processes,
            args=(processes,),
            daemon=True,
            name=f"random-workload-{source}"
        ).start()
        return [proc.pid for _, _, proc in processes]
    except Exception as e:
        print(f"❌ 触发随机负载失败：{str(e)}")
        return []


def sample_process_state(pid: int) -> Dict[str, Any]:
    """采集 ps/lscpu + workload(pid) 维度的 perf/top 采样"""
    samples: Dict[str, Any] = {}

    # 与基线采样保持一致：全量ps/lscpu + per-pid perf/top
    samples["ps_ef"] = execute_shell_command(["ps", "-ef"], timeout=5)
    samples["lscpu"] = execute_shell_command(["lscpu"], timeout=5)
    workload_processes = get_workload_processes()
    workload_pids = [item["pid"] for item in workload_processes]
    samples["workload_processes"] = workload_processes
    samples["workload_l3_hit_rate"] = sample_workload_l3_hit_rate(workload_pids, sample_seconds=0.5)
    samples["workload_cpu_percent"] = sample_workload_cpu_percent_top(workload_pids, delay_seconds=0.2, iterations=2)
    # 返回所有负载进程的最新采样日志，而非仅目标PID
    samples["benchmark_latest"] = collect_latest_benchmark_samples(workload_pids)
    return samples


def run_single_bind_command(command_str: str) -> BindCommandResult:
    """执行单条绑核指令，等待1秒后采样，再解除绑核"""
    result = BindCommandResult(
        command=command_str,
        pid=None,
        bind_success=False,
        sample_results={},
        exit_code=-1,
        reward=None,
        error_msg=""
    )
    proc: Optional[subprocess.Popen] = None
    try:
        cmd_parts = shlex.split(command_str)
        if not cmd_parts:
            result.error_msg = "命令不能为空"
            result.reward = failure_reward(result.error_msg)
            return result

        base_cmd = cmd_parts[0]
        if base_cmd not in ALLOWED_BASE_COMMANDS:
            result.error_msg = f"禁止执行命令：{base_cmd}（仅允许{ALLOWED_BASE_COMMANDS}）"
            result.reward = failure_reward(result.error_msg)
            return result

        # 绑定对象必须是 *intensive 任务
        if base_cmd == "numactl" and not _is_intensive_command(cmd_parts):
            result.error_msg = "numactl 绑定的目标命令必须是 *intensive 二进制"
            result.reward = failure_reward(result.error_msg)
            return result

        sample_pid: Optional[int] = None
        if base_cmd == "taskset":
            for token in reversed(cmd_parts):
                if token.isdigit():
                    sample_pid = int(token)
                    break
            if sample_pid is None:
                result.error_msg = "taskset绑核缺少目标PID"
                result.reward = failure_reward(result.error_msg)
                return result
            if not _pid_is_intensive(sample_pid):
                result.error_msg = f"PID {sample_pid} 不是 *intensive 结尾的任务，拒绝绑核"
                result.reward = failure_reward(result.error_msg)
                return result

        # reward baseline：先采一份“执行前”的最新 benchmark 日志
        workload_processes = get_workload_processes()
        workload_pids = [item["pid"] for item in workload_processes]
        before_latest = collect_latest_benchmark_samples(workload_pids)

        proc = subprocess.Popen(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="ignore"
        )
        register_tracked_process(proc, source=f"bind_task:{base_cmd}")
        result.bind_success = True
        result.pid = sample_pid if sample_pid is not None else proc.pid

        # taskset 属于短命令：等待其结束并检查退出码，失败则直接返回 reward=-1.0
        if base_cmd == "taskset":
            stdout_text, stderr_text = proc.communicate(timeout=5)
            result.exit_code = proc.returncode if proc.returncode is not None else -1
            if result.exit_code != 0:
                result.bind_success = False
                result.error_msg = (stderr_text or stdout_text or f"taskset 执行失败：exit_code={result.exit_code}").strip()
                result.reward = failure_reward(result.error_msg)
                return result

        # 运行1秒后采样
        time.sleep(1)
        result.sample_results = sample_process_state(result.pid or 0)

        # 对于非 taskset 命令：若已退出且返回码非0，视为执行失败
        proc.poll()
        if proc.returncode is not None:
            result.exit_code = proc.returncode
            if proc.returncode != 0:
                stdout_text, stderr_text = proc.communicate(timeout=2)
                result.bind_success = False
                result.error_msg = (stderr_text or stdout_text or f"{base_cmd} 执行失败：exit_code={proc.returncode}").strip()
                result.reward = failure_reward(result.error_msg)
                return result

        after_latest = {}
        if isinstance(result.sample_results, dict):
            after_latest = result.sample_results.get("benchmark_latest", {}) or {}
        if isinstance(after_latest, dict):
            result.reward = compute_ops_change_rate_reward(before_latest, after_latest)
        else:
            result.reward = failure_reward("benchmark_latest 缺失，无法计算 reward")

        if proc and proc.returncode is None:
            result.exit_code = 0
    except Exception as e:
        result.error_msg = f"命令执行异常：{str(e)}\n{traceback.format_exc()}"
        if result.reward is None:
            result.reward = failure_reward(result.error_msg)
        if proc and proc.poll() is None:
            proc.kill()
    finally:
        cleanup_finished_processes()
    return result


def process_bind_task(task_params: Dict[str, Any]) -> BindTaskResult:
    """处理一串绑核指令：按顺序执行并采样，返回聚合结果"""
    request_id = task_params["request_id"]
    commands: List[str] = task_params["bind_commands"]

    command_results: List[BindCommandResult] = []
    success = True
    error_msg = ""

    try:
        for cmd in commands:
            single_result = run_single_bind_command(cmd)
            if not single_result.bind_success and single_result.reward is None:
                single_result.reward = failure_reward(single_result.error_msg or "命令执行失败")
            command_results.append(single_result)
            if not single_result.bind_success:
                success = False
                if not error_msg:
                    error_msg = single_result.error_msg
    except Exception as e:
        success = False
        error_msg = f"任务执行异常：{str(e)}\n{traceback.format_exc()}"

    task_reward = command_results[-1].reward if command_results else None
    if not success:
        task_reward = failure_reward(error_msg or "任务执行失败")
    return BindTaskResult(
        request_id=request_id,
        success=success,
        command_results=command_results,
        reward=task_reward,
        error_msg=error_msg
    )

def process_queue():
    """处理任务队列（后台线程，串行执行）"""
    global is_processing
    with queue_lock:
        if is_processing or not task_queue:
            return
        is_processing = True

    try:
        while True:
            with queue_lock:
                if not task_queue:
                    break
                current_task = task_queue.pop(0)  # FIFO
                processing_requests.add(current_task["request_id"])

            try:
                result = process_bind_task(current_task)
            finally:
                with queue_lock:
                    processing_requests.discard(current_task["request_id"])

            # 存储结果（供调用方获取，这里简化为内存存储，生产环境可改用Redis/数据库）
            with queue_lock:
                completed_results[result.request_id] = result

    finally:
        with queue_lock:
            is_processing = False


@app.on_event("startup")
async def on_startup():
    """服务启动时直接触发benchmark负载"""
    trigger_random_workload_async(source="startup")

# ======================== API接口 ========================
@app.post("/bind-tasks")
async def submit_bind_tasks(
    request_id: str = Body(..., description="唯一请求ID"),
    bind_commands: List[str] = Body(..., description="一串绑核指令（按顺序执行）"),
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """提交绑核指令序列，放入队列按顺序执行（异步，立即返回）"""
    if x_api_key != AUTH_API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API Key错误")

    if not bind_commands:
        print("bind_commands不能为空")
        raise HTTPException(status_code=400, detail="bind_commands不能为空")

    with queue_lock:
        duplicate = (
            any(task["request_id"] == request_id for task in task_queue)
            or request_id in completed_results
            or request_id in processing_requests
        )
        if duplicate:
            print(f"请求ID{request_id}已存在，请勿重复提交")
            raise HTTPException(status_code=400, detail=f"请求ID{request_id}已存在，请勿重复提交")

        task_queue.append({"request_id": request_id, "bind_commands": bind_commands})
        queued_size = len(task_queue)

    threading.Thread(target=process_queue, daemon=True).start()

    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "msg": "任务已入队，等待执行",
            "data": {
                "request_id": request_id,
                "queue_size": queued_size
            }
        }
    )


@app.get("/bind-tasks/{request_id}")
async def query_bind_result(
    request_id: str,
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """根据request_id查询绑核采样结果（运行中/未找到/已完成）"""
    if x_api_key != AUTH_API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API Key错误")

    with queue_lock:
        if request_id in completed_results:
            result = completed_results[request_id]
            result_dict = asdict(result)
            reward_detail = result_dict.get("reward")
            reward_score: float
            if isinstance(reward_detail, dict) and isinstance(reward_detail.get("score"), (int, float)):
                reward_score = float(reward_detail["score"])
            else:
                reward_score = -1.0
            if not result.success:
                reward_score = -1.0
            return JSONResponse(
                status_code=200,
                content={
                    "code": 200 if result.success else 500,
                    "msg": result.error_msg if result.error_msg else "任务执行完成",
                    "data": result_dict,
                    "reward": reward_score,
                }
            )

        queued = any(task["request_id"] == request_id for task in task_queue)
        running = request_id in processing_requests

    if queued or running:
        return JSONResponse(
            status_code=202,
            content={
                "code": 202,
                "msg": "任务正在执行，请稍后查询",
                "data": {"request_id": request_id}
            }
        )

    raise HTTPException(status_code=404, detail=f"请求ID{request_id}不存在")


@app.post("/stop-all-processes")
async def stop_all_processes(
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """终止由当前server启动并仍在运行的所有进程"""
    if x_api_key != AUTH_API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API Key错误")

    summary = stop_all_tracked_processes()
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "msg": "已尝试停止所有server侧已记录的进程",
            "data": summary
        }
    )


@app.post("/start-random-workload")
async def start_random_workload(
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """
    再次随机启动一批benchmark负载，参数与服务启动时一致。
    任务在后台线程运行，立即返回。
    """
    if x_api_key != AUTH_API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API Key错误")

    pids = trigger_random_workload_async(source="api")
    pid_msg = ",".join(str(pid) for pid in pids)
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "msg": pid_msg
        }
    )

@app.get("/baseline-sample")
async def baseline_sample(
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """返回当前机器的初始 ps/lscpu + workload(pid) 维度的 perf/top 采样结果"""
    if x_api_key != AUTH_API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API Key错误")
    samples = collect_baseline_sample()
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "msg": "基线采样完成",
            "data": samples
        }
    )

# 健康检查接口
@app.get("/health")
async def health_check():
    with queue_lock:
        queue_size = len(task_queue)
    return {
        "status": "ok",
        "service": "numa-bind-task-executor",
        "queue_size": queue_size,
        "is_processing": is_processing
    }

if __name__ == "__main__":
    import uvicorn
    # 启动FastAPI服务，监听所有网卡
    uvicorn.run(app, host="0.0.0.0", port=8000)
