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

# 初始化FastAPI应用
app = FastAPI(title="NUMA Bind Task Executor", version="1.0")

# ======================== 核心配置 ========================
# 1. API鉴权Key（调用方需携带）
AUTH_API_KEY = "container-a-secure-key-2025"

# 2. 安全配置：允许的基础命令（绑核/采样相关）
ALLOWED_BASE_COMMANDS = {
    "numactl", "ps", "lscpu", "perf", "taskset", "kill", "grep"
}

# 3. 任务队列（FIFO）+ 锁（保证线程安全）
task_queue: List[Dict[str, Any]] = []
queue_lock = threading.Lock()
is_processing = False
completed_results: Dict[str, Any] = {}
processing_requests = set()

# 4. NUMA/CPU合法性校验正则
NUMA_NODE_PATTERN = re.compile(r"^\d+$")  # 数字格式的NUMA节点
CPU_LIST_PATTERN = re.compile(r"^\d+(,\d+)*(-\d+)*$")  # 支持1,2,3 或 0-7格式
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
DISK_TEST_FILE = BENCHMARK_DIR / "disk_test.tmp"
TEST_DURATION = 60
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

# ======================== 数据结构定义 ========================
@dataclass
class BindCommandResult:
    """单条绑核指令的执行与采样结果"""
    command: str
    pid: Optional[int]
    bind_success: bool
    sample_results: Dict[str, Any]
    exit_code: int
    error_msg: str = ""

@dataclass
class BindTaskResult:
    """绑核任务（包含多条指令）的整体结果"""
    request_id: str
    success: bool
    command_results: List[BindCommandResult]
    error_msg: str = ""

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

def collect_baseline_sample() -> Dict[str, Any]:
    """采集当前机器的初始ps/lscpu/perf状态"""
    return {
        "ps_ef": execute_shell_command(["ps", "-ef"], timeout=5),
        "lscpu": execute_shell_command(["lscpu"], timeout=5),
        "perf_stat": execute_shell_command(
            ["perf", "stat", "sleep", "0.5"],
            timeout=6
        )
    }


def random_generate_load_counts() -> Dict[str, int]:
    load_counts: Dict[str, int] = {}
    for load_name, (min_cnt, max_cnt) in LOAD_COUNT_RANGE.items():
        load_counts[load_name] = random.randint(min_cnt, max_cnt)
    total = sum(load_counts.values())
    if total == 0:
        print("⚠️ 所有负载随机数量均为0，强制为compute类型分配1个实例")
        load_counts["compute"] = 1
    return load_counts


def _run_compute_load(duration: int) -> None:
    end = time.time() + duration
    x = 1
    while time.time() < end:
        x = (x * 3 + 7) % 10000019  # 简单CPU计算


def _run_mem_load(duration: int) -> None:
    end = time.time() + duration
    chunk = bytearray(MEM_GRANULARITY * 1024)
    pool = [chunk[:] for _ in range(min(MEM_THREADS, 16))]
    idx = 0
    while time.time() < end:
        pool[idx % len(pool)][0] = (pool[idx % len(pool)][0] + 1) % 256
        idx += 1


def _run_cache_load(duration: int) -> None:
    end = time.time() + duration
    data = [i for i in range(1024 * 16)]
    idx = 0
    while time.time() < end:
        data[idx % len(data)] ^= 1
        idx += 1


def _run_disk_load(duration: int) -> None:
    end = time.time() + duration
    try:
        with open(DISK_TEST_FILE, "wb") as f:
            block = b"0" * (DISK_BLOCK_SIZE_KB * 1024)
            while time.time() < end:
                f.write(block)
                f.flush()
                os.fsync(f.fileno())
    except Exception as e:
        print(f"⚠️ 磁盘负载异常：{e}")
    finally:
        if DISK_TEST_FILE.exists():
            try:
                os.remove(DISK_TEST_FILE)
            except Exception:
                pass


def start_thread_load_instances(load_counts: Dict[str, int]) -> List[threading.Thread]:
    threads: List[threading.Thread] = []
    total_instances = sum(load_counts.values())
    print(f"\n🚀 开始启动 {total_instances} 个benchmark任务线程：")
    def spawn(load_name: str, target_func):
        t = threading.Thread(target=target_func, args=(TEST_DURATION,), daemon=True)
        t.start()
        return t

    for load_name, count in load_counts.items():
        if count <= 0:
            print(f"  - {load_name}: 0 个实例（跳过）")
            continue
        print(f"  - {load_name}: {count} 个线程实例")
        target = {
            "compute": _run_compute_load,
            "mem": _run_mem_load,
            "cache": _run_cache_load,
            "disk": _run_disk_load
        }[load_name]
        for idx in range(1, count + 1):
            t = spawn(load_name, target)
            threads.append(t)
            print(f"    ✅ 已启动 {load_name}[{idx}] 线程")
    return threads


def wait_for_threads(threads: List[threading.Thread]) -> None:
    if not threads:
        return
    print(f"\n⌛ 等待所有benchmark线程运行 {TEST_DURATION} 秒...")
    start_time = time.time()
    for t in threads:
        t.join(timeout=TEST_DURATION + 5)
    elapsed = time.time() - start_time
    print(f"\n📊 所有benchmark线程运行完成，总耗时：{elapsed:.2f} 秒")


def start_benchmark_workload() -> None:
    """服务启动时直接在当前进程启动随机benchmark负载"""
    try:
        load_counts = random_generate_load_counts()
        print("\n📋 随机生成的benchmark实例数量：")
        for load_name, count in load_counts.items():
            print(f"  - {load_name}: {count} 个")
        threads = start_thread_load_instances(load_counts)
        wait_for_threads(threads)
        print("\n🎉 随机多线程benchmark负载完成！")
    except Exception as e:
        print(f"❌ benchmark 任务异常：{str(e)}")

def sample_process_state(pid: int) -> Dict[str, Any]:
    """采集指定进程的ps/lscpu/perf信息"""
    samples: Dict[str, Any] = {}

    samples["ps_ef"] = execute_shell_command(["ps", "-fp", str(pid)], timeout=5)

    samples["lscpu"] = execute_shell_command(["lscpu"], timeout=5)
    samples["perf_stat"] = execute_shell_command(
        ["perf", "stat", "-p", str(pid), "-o", "/dev/stdout", "sleep", "0.5"],
        timeout=6
    )
    return samples


def run_single_bind_command(command_str: str) -> BindCommandResult:
    """执行单条绑核指令，等待1秒后采样，再解除绑核"""
    result = BindCommandResult(
        command=command_str,
        pid=None,
        bind_success=False,
        sample_results={},
        exit_code=-1,
        error_msg=""
    )
    proc: Optional[subprocess.Popen] = None
    try:
        cmd_parts = shlex.split(command_str)
        if not cmd_parts:
            result.error_msg = "命令不能为空"
            return result

        base_cmd = cmd_parts[0]
        if base_cmd not in ALLOWED_BASE_COMMANDS:
            result.error_msg = f"禁止执行命令：{base_cmd}（仅允许{ALLOWED_BASE_COMMANDS}）"
            return result

        sample_pid: Optional[int] = None
        if base_cmd == "taskset":
            for token in reversed(cmd_parts):
                if token.isdigit():
                    sample_pid = int(token)
                    break
            if sample_pid is None:
                result.error_msg = "taskset绑核缺少目标PID"
                return result

        proc = subprocess.Popen(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="ignore"
        )
        result.bind_success = True
        result.pid = sample_pid if sample_pid is not None else proc.pid

        # 运行1秒后采样
        time.sleep(1)
        target_pid = result.pid
        if target_pid and (proc.poll() is None or base_cmd == "taskset"):
            result.sample_results = sample_process_state(target_pid)
        else:
            result.sample_results = {
                "ps_ef": {"exit_code": -1, "stdout": "", "stderr": "目标进程已退出，无法采样"},
                "lscpu": execute_shell_command(["lscpu"], timeout=5),
                "perf_stat": {"exit_code": -1, "stdout": "", "stderr": "目标进程已退出，无法采样"}
            }

        # 解绑：恢复CPU亲和性为全核，让进程自行结束
        if target_pid:
            cpu_cnt = os.cpu_count() or 1
            full_mask = hex((1 << cpu_cnt) - 1)
            unbind_res = execute_shell_command(["taskset", "-p", full_mask, str(target_pid)], timeout=5)
            result.sample_results["unbind_taskset"] = unbind_res

        result.exit_code = proc.returncode if proc else -1
    except Exception as e:
        result.error_msg = f"命令执行异常：{str(e)}\n{traceback.format_exc()}"
        if proc and proc.poll() is None:
            proc.kill()
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
            command_results.append(single_result)
            if not single_result.bind_success:
                success = False
                if not error_msg:
                    error_msg = single_result.error_msg
    except Exception as e:
        success = False
        error_msg = f"任务执行异常：{str(e)}\n{traceback.format_exc()}"

    return BindTaskResult(
        request_id=request_id,
        success=success,
        command_results=command_results,
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
    threading.Thread(target=start_benchmark_workload, daemon=True).start()

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
        raise HTTPException(status_code=400, detail="bind_commands不能为空")

    with queue_lock:
        duplicate = (
            any(task["request_id"] == request_id for task in task_queue)
            or request_id in completed_results
            or request_id in processing_requests
        )
        if duplicate:
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
            return JSONResponse(
                status_code=200,
                content={
                    "code": 200 if result.success else 500,
                    "msg": result.error_msg if result.error_msg else "任务执行完成",
                    "data": result_dict
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

@app.get("/baseline-sample")
async def baseline_sample(
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """返回当前机器的初始ps/lscpu/perf采样结果"""
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
