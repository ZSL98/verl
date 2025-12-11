#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import subprocess
import time
import os
import sys
from typing import List, Dict, Tuple

TEST_DURATION = 60
DISK_TEST_FILE = "./disk_test.tmp"

LOAD_COUNT_RANGE: Dict[str, Tuple[int, int]] = {
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

LOAD_DEFINITIONS: Dict[str, callable] = {
    "compute": lambda: [
        "./cpubench/compute_intensive",
        "-t", str(COMPUTE_THREADS),
        "-T", str(COMPUTE_THREADS),
        "-f", "0",
        "-d", "0",
        "-r", str(TEST_DURATION)
    ],
    "mem": lambda: [
        "./cpubench/mem_intensive",
        "-t", str(MEM_THREADS),
        "-T", str(MEM_THREADS),
        "-M", str(MEM_SIZE_MB),
        "-g", str(MEM_GRANULARITY),
        "-s", str(MEM_SEQUENTIAL),
        "-f", "0",
        "-d", "0",
        "-r", str(TEST_DURATION)
    ],
    "cache": lambda: [
        "./cpubench/cache_sensitive",
        "-t", str(CACHE_THREADS),
        "-T", str(CACHE_THREADS),
        "-C", str(CACHE_SIZE_MB),
        "-f", "0",
        "-d", "0",
        "-r", str(TEST_DURATION)
    ],
    "disk": lambda: [
        "./cpubench/io_disk_intensive",
        "-t", str(DISK_THREADS),
        "-T", str(DISK_THREADS),
        "-p", DISK_TEST_FILE,
        "-F", str(DISK_FILE_SIZE_MB),
        "-b", str(DISK_BLOCK_SIZE_KB),
        "-s", str(DISK_SEQUENTIAL),
        "-R", str(DISK_READ_ONLY),
        "-f", "0",
        "-d", "0",
        "-r", str(TEST_DURATION)
    ]
}


def check_dependencies() -> None:
    """检查benchmark可执行文件是否存在"""
    required_binaries = [
        "./cpubench/compute_intensive",
        "./cpubench/mem_intensive",
        "./cpubench/cache_sensitive",
        "./cpubench/io_disk_intensive"
    ]
    missing = [bin for bin in required_binaries if not os.path.exists(bin) or not os.access(bin, os.X_OK)]
    if missing:
        print(f"错误：缺少可执行文件或权限不足：{', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def random_generate_load_counts() -> Dict[str, int]:
    load_counts = {}
    
    for load_name, (min_cnt, max_cnt) in LOAD_COUNT_RANGE.items():
        load_counts[load_name] = random.randint(min_cnt, max_cnt)
    
    total_count = sum(load_counts.values())
    if total_count == 0:
        print("⚠️  所有负载随机数量均为0，强制为compute类型分配1个实例")
        load_counts["compute"] = 1
    
    return load_counts


def start_load_instances(load_counts: Dict[str, int]) -> List[Tuple[str, int, subprocess.Popen]]:
    processes = []
    total_instances = sum(load_counts.values())
    print(f"\n🚀 开始启动 {total_instances} 个任务实例（按类型随机分配）：")
    
    for load_name, count in load_counts.items():
        if count <= 0:
            print(f"  - {load_name}: 0 个实例（跳过）")
            continue
        
        print(f"  - {load_name}: {count} 个实例")
        for instance_idx in range(1, count + 1):
            try:
                cmd = LOAD_DEFINITIONS[load_name]()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True
                )
                processes.append((load_name, instance_idx, proc))
                print(f"    ✅ 已启动 {load_name}[{instance_idx}] (PID: {proc.pid})")
            except Exception as e:
                print(f"    ❌ 启动 {load_name}[{instance_idx}] 失败：{str(e)}", file=sys.stderr)
    
    return processes


def wait_for_processes(processes: List[Tuple[str, int, subprocess.Popen]]) -> None:
    if not processes:
        return
    
    print(f"\n⌛ 等待所有任务实例运行 {TEST_DURATION} 秒...")
    start_time = time.time()
    
    for load_name, instance_idx, proc in processes:
        try:
            proc.wait(timeout=TEST_DURATION + 5)
            exit_code = proc.returncode
            if exit_code == 0:
                print(f"✅ {load_name}[{instance_idx}] 运行完成 (退出码: {exit_code})")
            else:
                print(f"⚠️ {load_name}[{instance_idx}] 异常退出 (退出码: {exit_code})", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"⚠️ {load_name}[{instance_idx}] 运行超时，强制终止", file=sys.stderr)
            proc.kill()
    
    if os.path.exists(DISK_TEST_FILE):
        try:
            os.remove(DISK_TEST_FILE)
            print(f"\n🗑️  已清理磁盘测试文件：{DISK_TEST_FILE}")
        except Exception as e:
            print(f"⚠️  清理磁盘测试文件失败：{str(e)}", file=sys.stderr)
    
    elapsed = time.time() - start_time
    print(f"\n📊 所有任务实例运行完成，总耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
    running_processes = []
    try:
        check_dependencies()
        
        load_counts = random_generate_load_counts()
        print("\n📋 随机生成的任务实例数量：")
        for load_name, count in load_counts.items():
            print(f"  - {load_name}: {count} 个")
        
        running_processes = start_load_instances(load_counts)
        
        wait_for_processes(running_processes)
        
        print("\n🎉 随机多实例负载测试完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断，强制终止所有进程", file=sys.stderr)
        for _, _, proc in running_processes:
            if proc.poll() is None:
                proc.kill()
        if os.path.exists(DISK_TEST_FILE):
            os.remove(DISK_TEST_FILE)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}", file=sys.stderr)
        sys.exit(1)