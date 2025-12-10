import time
import uuid
import requests
import random
from pathlib import Path
from typing import List, Dict, Any

# 容器A的接口地址（因在同一网络，可直接用容器名访问）
SERVER_BASE_URL = "http://127.0.0.1:8000"
# 鉴权Key（需与容器A的AUTH_API_KEY一致）
API_KEY = "container-a-secure-key-2025"


def _headers() -> Dict[str, str]:
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }


def fetch_baseline_sample() -> Dict[str, Any]:
    """从server获取初始状态的ps/lscpu/perf结果"""
    url = f"{SERVER_BASE_URL}/baseline-sample"
    resp = requests.get(url, headers=_headers(), timeout=20)
    return resp.json()


def submit_bind_task(request_id: str, commands: List[str]) -> Dict[str, Any]:
    """提交绑核指令序列"""
    url = f"{SERVER_BASE_URL}/bind-tasks"
    payload = {"request_id": request_id, "bind_commands": commands}
    resp = requests.post(url, json=payload, headers=_headers(), timeout=20)
    return resp.json()


def query_bind_result(request_id: str) -> Dict[str, Any]:
    """查询绑核任务结果"""
    url = f"{SERVER_BASE_URL}/bind-tasks/{request_id}"
    resp = requests.get(url, headers=_headers(), timeout=20)
    return resp.json()


def print_sample_results(title: str, samples: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for name, result in samples.items():
        print(f"[{name}] exit_code={result.get('exit_code')}")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if stdout:
            print(f"stdout:\n{stdout}")
        if stderr:
            print(f"stderr:\n{stderr}")


def pick_intensive_pid_from_ps(ps_stdout: str) -> int:
    """从ps -ef输出中随机挑选一个 *intensive 结尾的进程PID"""
    candidates = []
    for line in ps_stdout.splitlines():
        tokens = line.split(None, 7)
        if len(tokens) < 8:
            continue
        pid_str = tokens[1]
        cmd = tokens[7]
        if not pid_str.isdigit():
            continue
        cmd_parts = cmd.split()
        if any(Path(part).name.endswith("intensive") for part in cmd_parts):
            candidates.append(int(pid_str))
    if not candidates:
        raise RuntimeError("未找到 *intensive 结尾的进程供绑核")
    return random.choice(candidates)


def main():
    # 1) 初始状态采样
    baseline = fetch_baseline_sample()
    print(f"基线采样响应：code={baseline.get('code')} msg={baseline.get('msg')}")
    if baseline.get("data"):
        print_sample_results("初始状态", baseline["data"])

    # 2) 从基线ps -ef中选择一个 *intensive 进程做绑核
    ps_stdout = baseline.get("data", {}).get("ps_ef", {}).get("stdout", "")
    target_pid = pick_intensive_pid_from_ps(ps_stdout)
    print(f"\n选中的目标PID：{target_pid}")

    request_id = f"client-{uuid.uuid4()}"
    bind_commands = [f"taskset -cp 0 {target_pid}"]
    submit_resp = submit_bind_task(request_id, bind_commands)
    print(f"\n提交结果：code={submit_resp.get('code')} msg={submit_resp.get('msg')} request_id={request_id}")

    # 3) 轮询获取采样结果
    for _ in range(10):
        time.sleep(1)
        query_resp = query_bind_result(request_id)
        code = query_resp.get("code")
        if code in (200, 500):
            print(f"\n查询结果：code={code} msg={query_resp.get('msg')}")
            data = query_resp.get("data", {})
            for idx, cmd_result in enumerate(data.get("command_results", []), start=1):
                print(f"\n--- 指令 {idx}: {cmd_result.get('command')} ---")
                print(f"bind_success={cmd_result.get('bind_success')}, exit_code={cmd_result.get('exit_code')}")
                print_sample_results("采样结果", cmd_result.get("sample_results", {}))
            break
        else:
            print(f"任务未完成，继续等待... (status code={code})")
    else:
        print("查询超时，未获取到结果")


if __name__ == "__main__":
    main()
