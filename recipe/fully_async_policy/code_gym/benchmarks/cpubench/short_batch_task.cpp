#include "load_common.h"

// 短批次任务：生命周期短、批次触发、调度开销敏感
struct Config : BaseConfig {
    int task_duration_ms;   // 单个短任务时长（毫秒）
    int task_interval_ms;   // 批次间隔（毫秒）
    int batch_size;         // 每批次任务数
};

struct Stats : BaseStats {
    atomic<uint64_t> batch_count = 0;  // 总批次数
    atomic<uint64_t> task_count = 0;   // 总任务数
};

// 单个短任务（计算密集型，模拟短时间CPU占用）
void short_task(Stats& stats) {
    double a = 3.14159, b = 2.71828;
    for (int i = 0; i < 10000; ++i) {
        a = a * b + sqrt(a);
        b = b + sin(a);
    }
    stats.task_count++;
}

// 批次任务控制器：定时触发一批短任务
void batch_task(int duration_ms, double load_factor, Config& config, Stats& stats) {
    auto start = high_resolution_clock::now();
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < duration_ms) {
        if (load_factor < 1.0 && uniform_real_distribution<>(0.0, 1.0)(rng) > load_factor) {
            this_thread::sleep_for(milliseconds(config.task_interval_ms));
            continue;
        }

        // 触发一批短任务
        vector<thread> task_threads;
        for (int i = 0; i < config.batch_size; ++i) {
            task_threads.emplace_back(short_task, ref(stats));
        }
        for (auto& t : task_threads) t.join();

        stats.batch_count++;
        this_thread::sleep_for(milliseconds(config.task_interval_ms));
    }

    stats.total_ops += stats.batch_count * config.batch_size;
    atomic_double_add(stats.total_cpu_time, duration_ms / 1000.0);
}

void parse_args(int argc, char* argv[], Config& config) {
    parse_base_args(argc, argv, config);
    config.load_name = "Short-Batch-Task";
    config.task_duration_ms = 10;    // 单个任务10ms
    config.task_interval_ms = 100;   // 每100ms一批
    config.batch_size = 10;          // 每批10个任务

    const char* short_opts = "D:I:B:";
    const option long_opts[] = {
        {"task-duration", required_argument, nullptr, 'D'},
        {"task-interval", required_argument, nullptr, 'I'},
        {"batch-size", required_argument, nullptr, 'B'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'D': config.task_duration_ms = max(1, stoi(optarg)); break;
            case 'I': config.task_interval_ms = max(10, stoi(optarg)); break;
            case 'B': config.batch_size = max(1, stoi(optarg)); break;
        }
    }

    cout << "========================================" << endl
         << config.load_name << " Load Config" << endl
         << "========================================" << endl
         << "Base Threads: " << config.base_threads << endl
         << "Max Threads: " << config.max_threads << endl
         << "Task Duration: " << config.task_duration_ms << "ms" << endl
         << "Batch Interval: " << config.task_interval_ms << "ms" << endl
         << "Batch Size: " << config.batch_size << endl
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
    load_controller(config, stats, batch_task, ref(config), ref(stats));

    // 最终报告
    double elapsed = config.runtime_sec;
    cout << "========================================" << endl
         << config.load_name << " Load Report" << endl
         << "========================================" << endl
         << "Total Runtime: " << elapsed << "s" << endl
         << "Average Threads: " << fixed << setprecision(1) << stats.current_threads << endl
         << "Total Batches: " << stats.batch_count << endl
         << "Total Tasks: " << stats.task_count << endl
         << "Average Tasks Per Second: " << stats.task_count / elapsed << endl
         << "Average CPU Usage: " << fixed << setprecision(1)
         << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "%" << endl
         << "========================================" << endl;

    return 0;
}