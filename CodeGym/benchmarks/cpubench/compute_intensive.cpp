#include "load_common.h"

// 扩展配置
struct Config : BaseConfig {
    int ops_per_iter;  // 每次循环运算次数（控制计算强度）
};

// 扩展统计
struct Stats : BaseStats {
    atomic<uint64_t> float_ops = 0;  // 浮点运算总数
};

// 计算密集型任务（高ALU占用）
void compute_task(int duration_ms, double load_factor, Config& config, Stats& stats) {
    const int local_ops_per_iter = config.ops_per_iter;
    double a = 3.1415926535, b = 2.7182818284, c = 1.4142135623, d = 0.5772156649;
    uint64_t local_ops = 0;
    uint64_t local_float_ops = 0;

    auto start = high_resolution_clock::now();
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < duration_ms) {
        if (load_factor < 1.0 && uniform_real_distribution<>(0.0, 1.0)(rng) > load_factor) {
            this_thread::yield();
            continue;
        }

        // 高强度浮点运算（ALU饱和）
        for (int i = 0; i < local_ops_per_iter; ++i) {
            a = (a * b) + sqrt(c * d);
            b = sin(a) * cos(b) + log(d + 1.0);
            c = pow(a, 0.333) + tan(b);
            d = (d + c) * exp(-a / 1000.0);
        }
        local_ops += local_ops_per_iter;
        local_float_ops += local_ops_per_iter * 4;  // 每次循环4次浮点运算
    }

    stats.total_ops += local_ops;
    stats.float_ops += local_float_ops;
    atomic_double_add(stats.total_cpu_time, duration_ms / 1000.0);
}

// 解析扩展参数
void parse_args(int argc, char* argv[], Config& config) {
    parse_base_args(argc, argv, config);
    config.load_name = "Compute-Intensive";
    config.ops_per_iter = 2000;  // 默认每次循环2000次运算

    // 扩展参数（仅当前程序）
    const char* short_opts = "o:";
    const option long_opts[] = {
        {"ops-per-iter", required_argument, nullptr, 'o'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        if (opt == 'o') config.ops_per_iter = max(100, stoi(optarg));
    }

    // 打印配置
    cout << "========================================" << endl
         << config.load_name << " Load Config" << endl
         << "========================================" << endl
         << "Base Threads: " << config.base_threads << endl
         << "Max Threads: " << config.max_threads << endl
         << "Ops Per Iter: " << config.ops_per_iter << endl
         << "Fluctuation: " << config.load_fluctuation << "%" << endl
         << "Runtime: " << config.runtime_sec << "s" << endl
         << "Dynamic Adjust: " << (config.dynamic_adjust ? "Enabled" : "Disabled") << endl
         << "========================================" << endl;
}

int main(int argc, char* argv[]) {
    Config config;
    Stats stats;
    parse_args(argc, argv, config);

    // 启动负载控制器
    load_controller(config, stats, compute_task, ref(config), ref(stats));

    // 最终报告
    double elapsed = config.runtime_sec;
    cout << "========================================" << endl
         << config.load_name << " Load Report" << endl
         << "========================================" << endl
         << "Total Runtime: " << elapsed << "s" << endl
         << "Average Threads: " << fixed << setprecision(1) << stats.current_threads << endl
         << "Total Compute Ops: " << stats.total_ops / 1e6 << "M" << endl
         << "Total Float Ops: " << stats.float_ops / 1e9 << "B" << endl
         << "Average Throughput: " << (stats.float_ops / 1e6) / elapsed << "Mops/s" << endl
         << "Average CPU Usage: " << fixed << setprecision(1)
         << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "%" << endl
         << "========================================" << endl;

    return 0;
}