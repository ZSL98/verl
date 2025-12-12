#include "load_common.h"

struct Config : BaseConfig {
    size_t mem_size_mb;        // 内存缓冲区大小（MB）
    int mem_access_granularity;// 访问粒度（字节）
    bool sequential;           // 顺序访问（true）/随机访问（false）
};

struct Stats : BaseStats {
    atomic<uint64_t> mem_bytes = 0;  // 总内存访问字节数
    atomic<uint64_t> cache_misses_est = 0;  // 缓存缺失估算
};

void mem_task(int duration_ms, double load_factor, Config& config, Stats& stats, vector<char>& mem_buf) {
    const int ops_per_iter = 1500;
    size_t buf_size = mem_buf.size();
    int granularity = config.mem_access_granularity;
    uint64_t local_ops = 0;
    uint64_t local_mem_bytes = 0;
    uint64_t local_cache_misses = 0;

    uniform_int_distribution<> addr_dist(0, (buf_size - granularity) / granularity);
    size_t seq_idx = 0;

    auto start = high_resolution_clock::now();
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < duration_ms) {
        if (load_factor < 1.0 && uniform_real_distribution<>(0.0, 1.0)(rng) > load_factor) {
            this_thread::yield();
            continue;
        }

        for (int i = 0; i < ops_per_iter; ++i) {
            size_t idx;
            if (config.sequential) {
                idx = (seq_idx++) % ((buf_size - granularity) / granularity) * granularity;
            } else {
                idx = addr_dist(rng) * granularity;
            }

            // 读-改-写操作，确保内存访问有效
            mem_buf[idx] ^= 0xAA;
            mem_buf[(idx + granularity/2) % buf_size] += mem_buf[idx];
            mem_buf[(idx + granularity) % buf_size] = ~mem_buf[idx];

            local_ops++;
            local_mem_bytes += granularity * 3;  // 3次访问/操作
            // 缓存缺失估算
            if (!config.sequential || granularity > 64) local_cache_misses++;
        }
    }

    stats.total_ops += local_ops;
    stats.mem_bytes += local_mem_bytes;
    stats.cache_misses_est += local_cache_misses;
    atomic_double_add(stats.total_cpu_time, duration_ms / 1000.0);
}

void parse_args(int argc, char* argv[], Config& config) {
    parse_base_args(argc, argv, config);
    config.load_name = "Memory-Intensive";
    config.mem_size_mb = 256;
    config.mem_access_granularity = 64;
    config.sequential = false;

    const char* short_opts = "M:g:s:";
    const option long_opts[] = {
        {"mem-size", required_argument, nullptr, 'M'},
        {"granularity", required_argument, nullptr, 'g'},
        {"sequential", required_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'M': config.mem_size_mb = max(1, stoi(optarg)); break;
            case 'g': config.mem_access_granularity = max(1, stoi(optarg)); break;
            case 's': config.sequential = (stoi(optarg) != 0); break;
        }
    }

    cout << "========================================" << endl
         << config.load_name << " Load Config" << endl
         << "========================================" << endl
         << "Base Threads: " << config.base_threads << endl
         << "Max Threads: " << config.max_threads << endl
         << "Mem Size: " << config.mem_size_mb << "MB" << endl
         << "Access Granularity: " << config.mem_access_granularity << "B" << endl
         << "Access Mode: " << (config.sequential ? "Sequential" : "Random") << endl
         << "Fluctuation: " << config.load_fluctuation << "%" << endl
         << "Runtime: " << config.runtime_sec << "s" << endl
         << "Dynamic Adjust: " << (config.dynamic_adjust ? "Enabled" : "Disabled") << endl
         << "========================================" << endl;
}

int main(int argc, char* argv[]) {
    Config config;
    Stats stats;
    parse_args(argc, argv, config);

    // 初始化内存缓冲区
    size_t buf_size = config.mem_size_mb * 1024 * 1024;
    vector<char> mem_buf(buf_size);
    generate(mem_buf.begin(), mem_buf.end(), []() { return uniform_int_distribution<>(0, 255)(rng); });
    cout << "Allocated " << config.mem_size_mb << "MB memory buffer" << endl;

    // 启动负载控制器
    load_controller(config, stats, mem_task, ref(config), ref(stats), ref(mem_buf));

    // 最终报告
    double elapsed = config.runtime_sec;
    double total_gb = stats.mem_bytes / 1e9;
    cout << "========================================" << endl
         << config.load_name << " Load Report" << endl
         << "========================================" << endl
         << "Total Runtime: " << elapsed << "s" << endl
         << "Average Threads: " << fixed << setprecision(1) << stats.current_threads << endl
         << "Total Mem Ops: " << stats.total_ops / 1e6 << "M" << endl
         << "Total Mem Access: " << fixed << setprecision(2) << total_gb << "GB" << endl
         << "Average Bandwidth: " << fixed << setprecision(2) << (total_gb / elapsed) << "GB/s" << endl
         << "Estimated Cache Misses: " << stats.cache_misses_est / 1e6 << "M" << endl
         << "Average CPU Usage: " << fixed << setprecision(1)
         << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "%" << endl
         << "========================================" << endl;

    return 0;
}