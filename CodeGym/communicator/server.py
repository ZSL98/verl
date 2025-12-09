from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.responses import JSONResponse
import subprocess
import traceback
import shlex
import time
import re
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

def sample_process_state(pid: int) -> Dict[str, Any]:
    """采集指定进程的ps/lscpu/perf信息"""
    samples: Dict[str, Any] = {}

    ps_cmd = f"ps -ef | grep {pid} | grep -v grep"
    try:
        ps_result = subprocess.run(
            ps_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            encoding="utf-8",
            errors="ignore"
        )
        samples["ps_ef"] = {
            "exit_code": ps_result.returncode,
            "stdout": ps_result.stdout,
            "stderr": ps_result.stderr
        }
    except Exception as e:
        samples["ps_ef"] = {
            "exit_code": -2,
            "stdout": "",
            "stderr": f"采样失败：{str(e)}"
        }

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

        proc = subprocess.Popen(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="ignore"
        )
        result.bind_success = True
        result.pid = proc.pid

        # 运行1秒后采样
        time.sleep(1)
        result.sample_results = sample_process_state(proc.pid)

        # 解除绑核：确保进程结束
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)

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
