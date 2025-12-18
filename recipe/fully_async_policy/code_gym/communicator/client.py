import random
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


SERVER_BASE_URL = "http://127.0.0.1:8000"
API_KEY = "container-a-secure-key-2025"


class CodeGymClient:
    _instance: Optional["CodeGymClient"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "CodeGymClient":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        server_base_url: str = SERVER_BASE_URL,
        api_key: str = API_KEY,
        session: Optional[requests.Session] = None,
        health_timeout: int = 5,
    ) -> None:
        """
        初始化客户端，并与同目录下的 server.py 提供的服务做一次健康检查连接。
        """
        if getattr(self, "_initialized", False):
            self._validate_singleton_args(server_base_url, api_key, session, health_timeout)
            return

        with self.__class__._instance_lock:
            if getattr(self, "_initialized", False):
                self._validate_singleton_args(server_base_url, api_key, session, health_timeout)
                return

            self.server_base_url = server_base_url.rstrip("/")
            self.api_key = api_key
            self.session = session or requests.Session()
            self.session.headers.update(
                {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                }
            )
            self.health_timeout = health_timeout
            self._workload_pid_lock = threading.Lock()
            self._latest_started_workload_pids: List[int] = []
            self._latest_seen_workload_pids: List[int] = []
            self._ensure_server_ready()
            self._initialized = True

    def __copy__(self) -> "CodeGymClient":
        return self

    def __deepcopy__(self, memo) -> "CodeGymClient":
        return self

    def _validate_singleton_args(
        self,
        server_base_url: str,
        api_key: str,
        session: Optional[requests.Session],
        health_timeout: int,
    ) -> None:
        normalized_url = server_base_url.rstrip("/")
        if normalized_url != self.server_base_url:
            raise RuntimeError(
                "CodeGymClient 是单例：已初始化完成；后续请直接调用 CodeGymClient() 获取实例，或确保参数一致。"
            )
        if api_key != self.api_key:
            raise RuntimeError(
                "CodeGymClient 是单例：已初始化完成；后续请直接调用 CodeGymClient() 获取实例，或确保参数一致。"
            )
        if health_timeout != self.health_timeout:
            raise RuntimeError(
                "CodeGymClient 是单例：已初始化完成；后续请直接调用 CodeGymClient() 获取实例，或确保参数一致。"
            )
        if session is not None and session is not self.session:
            raise RuntimeError(
                "CodeGymClient 是单例：已初始化完成；后续请直接调用 CodeGymClient() 获取实例，或确保参数一致。"
            )

    def _headers(self) -> Dict[str, str]:
        return dict(self.session.headers)

    def _ensure_server_ready(self) -> None:
        health_url = f"{self.server_base_url}/health"
        try:
            resp = self.session.get(health_url, timeout=self.health_timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"无法连接到 CodeGym server：{health_url}，请确认 server.py 已启动"
            ) from exc

    def fetch_baseline_sample(self) -> Dict[str, Any]:
        url = f"{self.server_base_url}/baseline-sample"
        resp = self.session.get(url, headers=self._headers(), timeout=20)
        payload: Dict[str, Any] = resp.json()
        if payload.get("code") == 200:
            data = payload.get("data", {}) or {}
            workload_processes = data.get("workload_processes", []) or []
            pids = self._parse_workload_pids(workload_processes)
            with self._workload_pid_lock:
                self._latest_seen_workload_pids = pids
        return payload

    def submit_bind_task(self, request_id: str, commands: List[str]) -> Dict[str, Any]:
        url = f"{self.server_base_url}/bind-tasks"
        payload = {"request_id": request_id, "bind_commands": commands}
        resp = self.session.post(url, json=payload, headers=self._headers(), timeout=20)
        return resp.json()

    def query_bind_result(self, request_id: str) -> Dict[str, Any]:
        """查询绑核任务结果"""
        url = f"{self.server_base_url}/bind-tasks/{request_id}"
        resp = self.session.get(url, headers=self._headers(), timeout=20)
        return resp.json()

    def stop_all_processes(self) -> Dict[str, Any]:
        """请求server终止其已记录的运行进程"""
        url = f"{self.server_base_url}/stop-all-processes"
        resp = self.session.post(url, headers=self._headers(), timeout=10)
        payload: Dict[str, Any] = resp.json()
        if payload.get("code") == 200:
            with self._workload_pid_lock:
                self._latest_started_workload_pids = []
                self._latest_seen_workload_pids = []
        return payload

    def start_random_workload(self) -> Dict[str, Any]:
        """请求server再启动一批随机benchmark负载（与启动时配置一致）"""
        url = f"{self.server_base_url}/start-random-workload"
        resp = self.session.post(url, headers=self._headers(), timeout=10)
        payload: Dict[str, Any] = resp.json()
        if payload.get("code") == 200:
            pid_msg = str(payload.get("msg", "") or "")
            pids = self._parse_pid_msg(pid_msg)
            with self._workload_pid_lock:
                self._latest_started_workload_pids = pids
            data = payload.get("data")
            if not isinstance(data, dict):
                data = {}
                payload["data"] = data
            data["pids"] = pids
        return payload

    def _parse_pid_msg(self, pid_msg: str) -> List[int]:
        pids: List[int] = []
        if not pid_msg:
            return pids
        for part in pid_msg.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                pids.append(int(part))
            except ValueError:
                continue
        return list(dict.fromkeys(pids))

    def _parse_workload_pids(self, workload_processes: Any) -> List[int]:
        if not isinstance(workload_processes, list):
            return []
        pids: List[int] = []
        for item in workload_processes:
            if not isinstance(item, dict):
                continue
            pid = item.get("pid")
            try:
                pids.append(int(pid))
            except (TypeError, ValueError):
                continue
        return list(dict.fromkeys(pids))

    def get_tracked_workload_pids(self) -> List[int]:
        with self._workload_pid_lock:
            pids = self._latest_seen_workload_pids + self._latest_started_workload_pids
        return list(dict.fromkeys(pids))

    def pick_random_workload_pid(self, workload_processes: Optional[List[Dict[str, Any]]] = None) -> int:
        """随机挑选一个由 server 启动的 workload PID。"""
        if workload_processes is not None:
            candidates = self._parse_workload_pids(workload_processes)
        else:
            candidates = self.get_tracked_workload_pids()
        if not candidates:
            raise RuntimeError("未找到可用的 workload 进程 PID 供绑核（请先确保 server 已启动负载）")
        return random.choice(candidates)

    def print_sample_results(self, title: str, samples: Dict[str, Any]) -> None:
        print(f"\n=== {title} ===")
        for name, result in samples.items():
            if name == "workload_processes" and isinstance(result, list):
                print(f"[{name}] count={len(result)}")
                for item in result:
                    pid = item.get("pid")
                    source = item.get("source", "")
                    command = item.get("command", "")
                    print(f"  - pid={pid} source={source} cmd={command}")
                continue

            if name == "workload_l3_hit_rate" and isinstance(result, dict):
                print(f"[{name}] exit_code={result.get('exit_code')} loads={result.get('loads_event')} misses={result.get('misses_event')}")
                results = result.get("results", {}) or {}
                for pid, metrics in sorted(results.items(), key=lambda it: int(it[0]) if str(it[0]).isdigit() else str(it[0])):
                    hit_rate = metrics.get("hit_rate")
                    loads = metrics.get("loads")
                    misses = metrics.get("misses")
                    exit_code = metrics.get("exit_code")
                    attempt = metrics.get("attempt")
                    sample_seconds = metrics.get("sample_seconds")
                    retry_note = ""
                    if attempt and attempt != 1:
                        retry_note = f" attempt={attempt} sample_seconds={sample_seconds}"
                    print(f"  - pid={pid} hit_rate={hit_rate} loads={loads} misses={misses} exit_code={exit_code}{retry_note}")
                stderr = result.get("stderr", "")
                if stderr:
                    print(f"stderr:\n{stderr}")
                continue

            if name == "workload_cpu_percent" and isinstance(result, dict):
                print(f"[{name}] exit_code={result.get('exit_code')}")
                cpu_percent = result.get("cpu_percent", {}) or {}
                for pid, value in sorted(cpu_percent.items(), key=lambda it: int(it[0]) if str(it[0]).isdigit() else str(it[0])):
                    print(f"  - pid={pid} cpu_percent={value}")
                stderr = result.get("stderr", "")
                if stderr:
                    print(f"stderr:\n{stderr}")
                continue

            if isinstance(result, dict) and {"exit_code", "stdout", "stderr"}.issubset(result.keys()):
                print(f"[{name}] exit_code={result.get('exit_code')}")
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                if stdout:
                    print(f"stdout:\n{stdout}")
                if stderr:
                    print(f"stderr:\n{stderr}")
                continue

            print(f"[{name}] {result}")

def main() -> None:
    """
    执行示例流程：
    1) 采集基线 ps/lscpu/perf 结果并打印
    2) 从 server 返回的 workload_processes 中随机挑选一个负载进程PID，生成 taskset 绑核指令
    3) 提交绑核任务并轮询查询结果，最终打印采样输出
    """
    client = CodeGymClient()

    baseline = client.fetch_baseline_sample()
    print(f"基线采样响应：code={baseline.get('code')} msg={baseline.get('msg')}")
    if baseline.get("data"):
        client.print_sample_results("初始状态", baseline["data"])

    try:
        workload_processes = baseline.get("data", {}).get("workload_processes", []) or []
        target_pid = client.pick_random_workload_pid(workload_processes)
    except RuntimeError as exc:
        print(f"选取目标PID失败：{exc}")
        return
    print(f"\n选中的目标PID：{target_pid}")

    request_id = f"client-{uuid.uuid4()}"
    bind_commands = [f"taskset -cp 0 {target_pid}"]
    submit_resp = client.submit_bind_task(request_id, bind_commands)
    print(f"\n提交结果：code={submit_resp.get('code')} msg={submit_resp.get('msg')} request_id={request_id}")

    for _ in range(10):
        time.sleep(1)
        query_resp = client.query_bind_result(request_id)
        code = query_resp.get("code")
        if code in (200, 500):
            print(f"\n查询结果：code={code} msg={query_resp.get('msg')}")
            data = query_resp.get("data", {}) or {}
            if data.get("reward") is not None:
                reward = data.get("reward") or {}
                print(f"\nreward_score={reward.get('score')} exit_code={reward.get('exit_code')}")
                if reward.get("stderr"):
                    print(f"reward_stderr:\n{reward.get('stderr')}")
            for idx, cmd_result in enumerate(data.get("command_results", []), start=1):
                print(f"\n--- 指令 {idx}: {cmd_result.get('command')} ---")
                print(f"bind_success={cmd_result.get('bind_success')}, exit_code={cmd_result.get('exit_code')}")
                if cmd_result.get("reward") is not None:
                    cmd_reward = cmd_result.get("reward") or {}
                    print(f"cmd_reward_score={cmd_reward.get('score')} exit_code={cmd_reward.get('exit_code')}")
                client.print_sample_results("采样结果", cmd_result.get("sample_results", {}))
            break
        print(f"任务未完成，继续等待... (status code={code})")
    else:
        print("查询超时，未获取到结果")

if __name__ == "__main__":
    main()
