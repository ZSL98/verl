#ifndef LOAD_COMMON_H
#define LOAD_COMMON_H

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cstdlib>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

using namespace std;
using namespace chrono;

// 全局配置结构体（每个程序可扩展）
struct BaseConfig {
    int base_threads;          // 基础线程数
    int max_threads;           // 最大线程数（动态调整时）
    int load_fluctuation;      // 负载波动幅度（0-100）
    int runtime_sec;           // 运行时间（秒）
    bool dynamic_adjust;       // 是否动态调整线程数
    string load_name;          // 负载名称（用于输出）
};

// 全局统计结构体（每个程序可扩展）
struct BaseStats {
    atomic<uint64_t> total_ops = 0;      // 总操作数
    atomic<double> total_cpu_time = 0;   // 总CPU时间（秒）
    atomic<int> current_threads = 0;     // 当前活跃线程数
    mutex stats_mtx;                     // 统计互斥锁
};

// 线程局部随机数生成器
thread_local mt19937 rng(random_device{}());

inline std::filesystem::path get_codegym_sample_dir() {
    const char* env_dir = std::getenv("CODEGYM_SAMPLE_DIR");
    if (env_dir && *env_dir) {
        return std::filesystem::path(env_dir);
    }
    return std::filesystem::temp_directory_path() / "codegym_samples";
}

inline std::filesystem::path get_codegym_sample_path() {
    static std::filesystem::path dir = get_codegym_sample_dir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir / ("sample_" + std::to_string(getpid()) + ".log");
}

inline void write_codegym_latest_sample(const std::string& line) {
    static std::filesystem::path path = get_codegym_sample_path();
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) return;
    out << line << std::endl;
}

inline void atomic_double_add(atomic<double>& target, double value) {
    double expected = target.load(memory_order_relaxed);
    while (!target.compare_exchange_weak(expected, expected + value,
                                         memory_order_relaxed, memory_order_relaxed)) {}
}

// 负载控制器基础逻辑（动态调整线程数和负载强度）
template <typename TaskFunc, typename... TaskArgs>
void load_controller(BaseConfig& config, BaseStats& stats, TaskFunc task_func, TaskArgs&&... args) {
    uniform_int_distribution<> thread_dist(config.base_threads, config.max_threads);
    uniform_real_distribution<> load_dist(0.3, 1.0);
    uniform_int_distribution<> fluct_dist(0, config.load_fluctuation);

    vector<thread> worker_threads;
    auto start_time = high_resolution_clock::now();

    // 初始化基础线程
    stats.current_threads = config.base_threads;
    for (int i = 0; i < config.base_threads; ++i) {
        worker_threads.emplace_back(task_func, 200, 1.0, forward<TaskArgs>(args)...);
    }

    // 动态调整循环
    while (duration_cast<seconds>(high_resolution_clock::now() - start_time).count() < config.runtime_sec) {
        this_thread::sleep_for(microseconds(200));
        double fluct_coeff = 1.0 + (fluct_dist(rng) - config.load_fluctuation / 2) / 100.0;
        double current_load = clamp(load_dist(rng) * fluct_coeff, 0.1, 1.0);

        // 动态调整线程数
        if (config.dynamic_adjust) {
            int target_threads = thread_dist(rng);
            // 移除多余线程
            while (stats.current_threads > target_threads && !worker_threads.empty()) {
                if (worker_threads.back().joinable()) {
                    worker_threads.back().join();
                    worker_threads.pop_back();
                    stats.current_threads--;
                }
            }
            // 添加新线程
            while (stats.current_threads < target_threads) {
                worker_threads.emplace_back(task_func, 200, current_load, forward<TaskArgs>(args)...);
                stats.current_threads++;
            }
        } else {
            // 固定线程数，调整负载强度
            for (auto& t : worker_threads) {
                if (t.joinable()) t.join();
                t = thread(task_func, 200, 0, forward<TaskArgs>(args)...);
            }
        }

        // 实时输出统计
        std::ostringstream oss;
        oss << "[" << config.load_name << " | " << fixed << setprecision(1) << elapsed << "s] "
            << "Threads: " << stats.current_threads << " | "
            << "Ops: " << stats.total_ops / 1e6 << "M | "
            << "CPU Usage(est): " << fixed << setprecision(1)
            << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "% | "
            << "Load Factor: " << fixed << setprecision(2) << current_load;
        const std::string line = oss.str();
        cout << line << endl;
        write_codegym_latest_sample(line);
    }

    // 等待所有线程结束
    for (auto& t : worker_threads) {
        if (t.joinable()) t.join();
    }
}

// 解析基础命令行参数
void parse_base_args(int argc, char* argv[], BaseConfig& config) {
    // 默认基础配置
    config.base_threads = 4;
    config.max_threads = 8;
    config.load_fluctuation = 20;
    config.runtime_sec = 60;
    config.dynamic_adjust = true;

    // 基础参数（所有程序共享）
    const char* short_opts = "t:T:f:r:d:";
    const option long_opts[] = {
        {"base-threads", required_argument, nullptr, 't'},
        {"max-threads", required_argument, nullptr, 'T'},
        {"fluctuation", required_argument, nullptr, 'f'},
        {"runtime", required_argument, nullptr, 'r'},
        {"dynamic", required_argument, nullptr, 'd'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 't': config.base_threads = stoi(optarg); break;
            case 'T': config.max_threads = stoi(optarg); break;
            case 'f': config.load_fluctuation = stoi(optarg); break;
            case 'r': config.runtime_sec = stoi(optarg); break;
            case 'd': config.dynamic_adjust = (stoi(optarg) != 0); break;
        }
    }

    // 参数校验
    config.base_threads = max(1, config.base_threads);
    config.max_threads = max(config.base_threads, config.max_threads);
    config.load_fluctuation = clamp(config.load_fluctuation, 0, 100);
    config.runtime_sec = max(1, config.runtime_sec);
}

#endif // LOAD_COMMON_H
