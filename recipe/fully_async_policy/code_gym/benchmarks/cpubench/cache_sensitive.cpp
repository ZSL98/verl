#include "load_common.h"

// 缓存敏感型：固定占用L3缓存，对缓存污染敏感
struct Config : BaseConfig {
    size_t cache_size_mb;  // 目标占用缓存大小（MB，需小于L3缓存）
};

struct Stats : BaseStats {
    atomic<uint64_t> cache_hits_est = 0;   // 缓存命中估算
    atomic<uint64_t> cache_misses_est = 0; // 缓存缺失估算
};

// 缓存敏感任务：循环访问固定大小的数组（确保高缓存命中）
void cache_task(int duration_ms, double load_factor, Config& config, Stats& stats, vector<int>& cache_buf) {
    const int ops_per_iter = 5000;
    // 调大单次统计粒度：将 1 个 op 定义为约 1000 次缓存微访问，降低 ops/s 数量级
    constexpr uint64_t kMicroOpsPerLogicalOp = 1000;
    const uint64_t logical_ops_per_iter = std::max<uint64_t>(
        1, static_cast<uint64_t>(ops_per_iter) / kMicroOpsPerLogicalOp
    );
    size_t buf_size = cache_buf.size();
    uint64_t local_ops = 0;
    uint64_t local_hits = 0;
    uint64_t local_misses = 0;

    auto start = high_resolution_clock::now();
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < duration_ms) {
        if (load_factor < 1.0 && uniform_real_distribution<>(0.0, 1.0)(rng) > load_factor) {
            this_thread::yield();
            continue;
        }

        for (int i = 0; i < ops_per_iter; ++i) {
            // 访问缓存缓冲区（高命中）
            size_t idx = i % buf_size;
            cache_buf[idx] = (cache_buf[idx] + 1) * 3;
            local_hits++;

            // 随机访问小概率外部内存（模拟缓存污染）
            if (uniform_int_distribution<>(0, 99)(rng) == 0) {
                int* temp = new int(uniform_int_distribution<>(0, 1000)(rng));
                *temp = *temp + cache_buf[idx];
                delete temp;
                local_misses++;
            }
        }
        local_ops += logical_ops_per_iter;
    }

    stats.total_ops += local_ops;
    stats.cache_hits_est += local_hits;
    stats.cache_misses_est += local_misses;
    atomic_double_add(stats.total_cpu_time, duration_ms / 1000.0);
}

void parse_args(int argc, char* argv[], Config& config) {
    parse_base_args(argc, argv, config);
    config.load_name = "Cache-Sensitive";
    config.cache_size_mb = 16;  // 默认占用16MB缓存（需根据CPU L3缓存调整）

    const char* short_opts = "C:";
    const option long_opts[] = {
        {"cache-size", required_argument, nullptr, 'C'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        if (opt == 'C') config.cache_size_mb = max(1, stoi(optarg));
    }

    // 提示：需根据CPU L3缓存调整（如CPU L3为32MB，建议cache-size<=24MB）
    cout << "========================================" << endl
         << config.load_name << " Load Config" << endl
         << "========================================" << endl
         << "Base Threads: " << config.base_threads << endl
         << "Max Threads: " << config.max_threads << endl
         << "Target Cache Size: " << config.cache_size_mb << "MB" << endl
         << "Fluctuation: " << config.load_fluctuation << "%" << endl
         << "Runtime: " << config.runtime_sec << "s" << endl
         << "Dynamic Adjust: " << (config.dynamic_adjust ? "Enabled" : "Disabled") << endl
         << "Note: Set cache-size <= 75% of CPU L3 cache" << endl
         << "========================================" << endl;
}

int main(int argc, char* argv[]) {
    Config config;
    Stats stats;
    parse_args(argc, argv, config);

    // 初始化缓存缓冲区（int数组：每个元素4字节，确保连续存储）
    size_t buf_size = (config.cache_size_mb * 1024 * 1024) / sizeof(int);
    vector<int> cache_buf(buf_size, 0);
    cout << "Allocated " << config.cache_size_mb << "MB cache buffer (" << buf_size << " ints)" << endl;

    // 启动负载控制器
    load_controller(config, stats, cache_task, ref(config), ref(stats), ref(cache_buf));

    // 最终报告
    double elapsed = config.runtime_sec;
    double hit_rate = (double)stats.cache_hits_est / (stats.cache_hits_est + stats.cache_misses_est) * 100;
    cout << "========================================" << endl
         << config.load_name << " Load Report" << endl
         << "========================================" << endl
         << "Total Runtime: " << elapsed << "s" << endl
         << "Average Threads: " << fixed << setprecision(1) << stats.current_threads << endl
         << "Total Ops: " << stats.total_ops / 1e6 << "M" << endl
         << "Cache Hits(est): " << stats.cache_hits_est / 1e6 << "M" << endl
         << "Cache Misses(est): " << stats.cache_misses_est / 1e3 << "K" << endl
         << "Cache Hit Rate(est): " << fixed << setprecision(1) << hit_rate << "%" << endl
         << "Average CPU Usage: " << fixed << setprecision(1)
         << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "%" << endl
         << "========================================" << endl;

    return 0;
}
