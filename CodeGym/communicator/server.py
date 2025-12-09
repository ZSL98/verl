from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.responses import JSONResponse
import subprocess
import traceback
import shlex
import time
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
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
task_queue: List[Dict] = []
queue_lock = threading.Lock()
is_processing = False

# 4. NUMA/CPU合法性校验正则
NUMA_NODE_PATTERN = re.compile(r"^\d+$")  # 数字格式的NUMA节点
CPU_LIST_PATTERN = re.compile(r"^\d+(,\d+)*(-\d+)*$")  # 支持1,2,3 或 0-7格式

# ======================== 数据结构定义 ========================
@dataclass
class TaskResult:
    """任务执行结果封装"""
    task_id: str
    bind_success: bool
    run_success: bool
    sample_results: Dict[str, str]
    exit_code: int
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

def process_numa_task(task_params: Dict) -> TaskResult:
    """
    处理NUMA绑核任务核心逻辑：
    1. 校验参数 → 2. 绑核启动任务 → 3. 运行1秒 → 4. 性能采样 → 5. 终止任务 → 6. 返回结果
    """
    task_id = task_params["task_id"]
    numa_node = task_params["numa_node"]
    cpu_list = task_params["cpu_list"]
    run_command = task_params["run_command"]
    timeout = task_params.get("timeout", 30)

    # 初始化结果
    task_result = TaskResult(
        task_id=task_id,
        bind_success=False,
        run_success=False,
        sample_results={},
        exit_code=-1,
        error_msg=""
    )

    try:
        # 步骤1：校验NUMA/CPU合法性
        if not validate_numa_cpu(numa_node, cpu_list):
            task_result.error_msg = f"非法参数：NUMA节点{numa_node}或CPU核心{cpu_list}不存在"
            return task_result

        # 步骤2：拆分目标运行命令（防止注入）
        run_cmd_parts = shlex.split(run_command)
        if not run_cmd_parts:
            task_result.error_msg = "运行命令不能为空"
            return task_result

        # 步骤3：构造绑核命令（numactl绑定NUMA节点+CPU核心）
        bind_cmd = [
            "numactl",
            f"--cpunodebind={numa_node}",
            f"--membind={numa_node}",
            f"--physcpubind={cpu_list}",
            *run_cmd_parts
        ]

        # 步骤4：启动绑核任务（后台运行）
        proc = None
        try:
            proc = subprocess.Popen(
                bind_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="ignore"
            )
            task_result.bind_success = True
            pid = proc.pid
            print(f"✅ 任务{task_id}：绑核成功，PID={pid}（NUMA{numa_node}, CPU{cpu_list}）")
        except Exception as e:
            task_result.error_msg = f"绑核启动失败：{str(e)}"
            return task_result

        # 步骤5：运行1秒（等待任务执行）
        time.sleep(1)

        # 步骤6：性能采样（ps/ lscpu/ perf stat）
        sample_cmds = {
            "ps_ef": ["ps", "-ef", "|", "grep", str(pid)],  # 进程信息
            "lscpu": ["lscpu"],  # 系统CPU/NUMA信息
            "perf_stat": ["perf", "stat", "-p", str(pid), "-o", "/dev/stdout", "sleep", "0.5"]  # 性能采样0.5秒
        }

        # 处理带管道的ps命令（特殊处理，临时允许shell=True）
        for sample_name, sample_cmd in sample_cmds.items():
            if sample_name == "ps_ef":
                # 管道命令需shell=True，加强参数过滤
                ps_cmd = f"ps -ef | grep {pid} | grep -v grep"
                try:
                    ps_result = subprocess.run(
                        ps_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5,
                        encoding="utf-8"
                    )
                    task_result.sample_results[sample_name] = {
                        "exit_code": ps_result.returncode,
                        "stdout": ps_result.stdout,
                        "stderr": ps_result.stderr
                    }
                except Exception as e:
                    task_result.sample_results[sample_name] = {
                        "exit_code": -2,
                        "stdout": "",
                        "stderr": f"采样失败：{str(e)}"
                    }
            else:
                # 普通命令（无shell注入风险）
                sample_result = execute_shell_command(sample_cmd, timeout=5)
                task_result.sample_results[sample_name] = sample_result

        # 步骤7：终止任务（取消绑核，任务结束）
        if proc.poll() is None:  # 进程仍在运行
            proc.terminate()
            proc.wait(timeout=2)
        task_result.run_success = True
        task_result.exit_code = 0
        task_result.error_msg = ""
        print(f"✅ 任务{task_id}：执行完成，已终止PID={pid}")

    except Exception as e:
        task_result.error_msg = f"任务执行异常：{str(e)}\n{traceback.format_exc()}"
        # 清理残留进程
        if 'proc' in locals() and proc and proc.poll() is None:
            proc.kill()

    return task_result

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

            # 处理当前任务
            result = process_numa_task(current_task)
            # 存储结果（供调用方获取，这里简化为内存存储，生产环境可改用Redis/数据库）
            current_task["result"] = result
            current_task["completed"] = True

    finally:
        with queue_lock:
            is_processing = False

# ======================== API接口 ========================
@app.post("/submit-numa-task")
async def submit_numa_task(
    # 结构化任务参数
    task_id: str = Body(..., description="唯一任务ID"),
    numa_node: str = Body(..., description="要绑定的NUMA节点（如0）"),
    cpu_list: str = Body(..., description="要绑定的CPU核心（如0-7或1,3,5）"),
    run_command: str = Body(..., description="要运行的目标命令（如./compute_intensive -t 1）"),
    timeout: int = Body(30, description="任务总超时时间（秒）"),
    x_api_key: str = Header(None, description="API鉴权Key")
):
    """
    提交NUMA绑核任务（阻塞式返回结果）
    流程：绑核→运行1秒→性能采样→终止任务→返回结果
    """
    # 1. 鉴权校验
    if x_api_key != AUTH_API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API Key错误")

    # 2. 校验任务ID唯一性
    with queue_lock:
        if any(task["task_id"] == task_id for task in task_queue):
            raise HTTPException(status_code=400, detail=f"任务ID{task_id}已存在，请勿重复提交")

    # 3. 构造任务参数
    task_params = {
        "task_id": task_id,
        "numa_node": numa_node,
        "cpu_list": cpu_list,
        "run_command": run_command,
        "timeout": timeout,
        "completed": False,
        "result": None
    }

    # 4. 加入任务队列
    with queue_lock:
        task_queue.append(task_params)

    # 5. 触发队列处理
    threading.Thread(target=process_queue, daemon=True).start()

    # 6. 阻塞等待任务完成（调用方阻塞）
    wait_start = time.time()
    while time.time() - wait_start < timeout:
        if task_params.get("completed", False):
            break
        time.sleep(0.1)  # 轮询间隔

    # 7. 检查超时
    if not task_params.get("completed", False):
        raise HTTPException(status_code=504, detail=f"任务{task_id}执行超时（{timeout}秒）")

    # 8. 返回结果
    result = task_params["result"]
    return JSONResponse(
        status_code=200,
        content={
            "code": 200 if result.exit_code == 0 else 500,
            "msg": result.error_msg if result.error_msg else "任务执行完成",
            "data": {
                "task_id": result.task_id,
                "bind_success": result.bind_success,
                "run_success": result.run_success,
                "exit_code": result.exit_code,
                "sample_results": result.sample_results
            }
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