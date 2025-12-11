#include "load_common.h"

struct Config : BaseConfig {
    string disk_path;       // 测试文件路径
    size_t file_size_mb;    // 测试文件大小（MB）
    int io_block_size_kb;   // IO块大小（KB）
    bool sequential;        // 顺序IO（true）/随机IO（false）
    bool read_only;         // 只读（true）/读写（false）
};

struct Stats : BaseStats {
    atomic<uint64_t> io_bytes = 0;  // 总IO字节数
    atomic<uint64_t> io_latency_us = 0;  // 总IO延迟（微秒）
};

// 磁盘IO任务（修复所有关键问题）
void disk_io_task(int duration_ms, double load_factor, Config& config, Stats& stats, int fd, size_t file_size) {
    const int block_size = config.io_block_size_kb * 1024;
    const int sector_size = 4096; // 系统页/扇区大小（根据实际环境调整）

    // ========== 修复1：O_DIRECT缓冲区对齐 ==========
    char* io_buf = nullptr;
    // 按sector_size对齐分配内存（posix_memalign是POSIX标准，兼容Linux/macOS）
    int align_ret = posix_memalign((void**)&io_buf, sector_size, block_size);
    if (align_ret != 0) {
        lock_guard<mutex> lock(stats.stats_mtx);
        cerr << "Error: posix_memalign failed (err=" << align_ret << "), block_size=" << block_size << endl;
        return;
    }
    // 复用随机数分布对象（修复6）
    static thread_local uniform_int_distribution<> byte_dist(0, 255);
    generate(io_buf, io_buf + block_size, []() { return byte_dist(rng); });

    // ========== 修复4：边界条件检查 ==========
    if (file_size < block_size) {
        lock_guard<mutex> lock(stats.stats_mtx);
        cerr << "Error: file_size(" << file_size << ") < block_size(" << block_size << ")" << endl;
        free(io_buf);
        return;
    }
    const size_t max_offset_idx = (file_size - block_size) / block_size;
    uniform_int_distribution<> offset_dist(0, max_offset_idx);
    // 复用负载因子随机分布（修复6）
    static thread_local uniform_real_distribution<> load_dist(0.0, 1.0);

    // 局部统计变量
    uint64_t local_ops = 0;
    uint64_t local_io_bytes = 0;
    uint64_t local_latency_us = 0;
    // 修复5：统计实际CPU时间（排除休眠）
    double local_cpu_time = 0.0;

    auto task_start = high_resolution_clock::now();
    auto start = high_resolution_clock::now();
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < duration_ms) {
        auto loop_start = high_resolution_clock::now();

        // 负载因子控制
        if (load_factor < 1.0 && load_dist(rng) > load_factor) {
            this_thread::sleep_for(milliseconds(1));
            continue;
        }

        // 计算IO偏移量
        off_t offset;
        if (config.sequential) {
            offset = (local_ops % max_offset_idx) * block_size;
        } else {
            offset = offset_dist(rng) * block_size;
        }

        // ========== 修复3：错误处理 - lseek ==========
        off_t lseek_ret = lseek(fd, offset, SEEK_SET);
        if (lseek_ret == -1) {
            lock_guard<mutex> lock(stats.stats_mtx);
            cerr << "Warning: lseek failed (offset=" << offset << "), err=" << strerror(errno) << endl;
            continue;
        }

        // 执行IO操作并统计延迟
        auto io_start = high_resolution_clock::now();
        ssize_t ret = -1;
        if (config.read_only) {
            ret = read(fd, io_buf, block_size);
        } else {
            // ========== 修复2：线程安全 - 写操作（可选：用pwrite/pread替代lseek+read/write） ==========
            // pread/pwrite 直接指定offset，无需lseek，天然线程安全
            ret = pwrite(fd, io_buf, block_size, offset);
        }
        auto io_end = high_resolution_clock::now();
        auto io_duration_us = duration_cast<microseconds>(io_end - io_start).count();

        // ========== 修复3：完整的IO结果处理 ==========
        if (ret == -1) {
            lock_guard<mutex> lock(stats.stats_mtx);
            cerr << "Warning: " << (config.read_only ? "read" : "write") 
                 << " failed (offset=" << offset << "), err=" << strerror(errno) << endl;
        } else if (ret == block_size) {
            local_ops++;
            local_io_bytes += block_size;
            local_latency_us += io_duration_us;
        } else {
            // 处理部分读写
            lock_guard<mutex> lock(stats.stats_mtx);
            cerr << "Warning: partial " << (config.read_only ? "read" : "write") 
                 << " (offset=" << offset << "), expected=" << block_size << ", actual=" << ret << endl;
        }

        // 累计实际CPU时间（修复5）
        auto loop_end = high_resolution_clock::now();
        local_cpu_time += duration_cast<microseconds>(loop_end - loop_start).count() / 1e6;
    }

    // 释放对齐的缓冲区（修复1）
    free(io_buf);

    // 更新全局统计
    stats.total_ops += local_ops;
    stats.io_bytes += local_io_bytes;
    stats.io_latency_us += local_latency_us;
    atomic_double_add(stats.total_cpu_time, local_cpu_time); // 修复5：用实际CPU时间更新
}

void parse_args(int argc, char* argv[], Config& config) {

    config.base_threads = 4;
    config.max_threads = 8;
    config.load_fluctuation = 20;
    config.runtime_sec = 60;
    config.dynamic_adjust = true;
    config.load_name = "Disk-IO-Intensive";
    config.disk_path = "./disk_test.tmp";
    config.file_size_mb = 1024;
    config.io_block_size_kb = 4;
    config.sequential = false;
    config.read_only = false;

    const char* short_opts = "t:T:f:r:d:p:F:b:s:R:";
    const option long_opts[] = {
        {"base-threads", required_argument, nullptr, 't'},
        {"max-threads", required_argument, nullptr, 'T'},
        {"fluctuation", required_argument, nullptr, 'f'},
        {"runtime", required_argument, nullptr, 'r'},
        {"dynamic", required_argument, nullptr, 'd'},
        {"path", required_argument, nullptr, 'p'},
        {"file-size", required_argument, nullptr, 'F'},
        {"block-size", required_argument, nullptr, 'b'},
        {"sequential", required_argument, nullptr, 's'},
        {"read-only", required_argument, nullptr, 'R'},
        {nullptr, 0, nullptr, 0}
    };

    int old_opterr = opterr;
    opterr = 0;

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 't': config.base_threads = stoi(optarg); break;
            case 'T': config.max_threads = stoi(optarg); break;
            case 'f': config.load_fluctuation = stoi(optarg); break;
            case 'r': config.runtime_sec = stoi(optarg); break;
            case 'd': config.dynamic_adjust = (stoi(optarg) != 0); break;
            case 'p': config.disk_path = optarg; break;
            case 'F': config.file_size_mb = max(10, stoi(optarg)); break;
            case 'b': config.io_block_size_kb = max(1, stoi(optarg)); break;
            case 's': config.sequential = (stoi(optarg) != 0); break;
            case 'R': config.read_only = (stoi(optarg) != 0); break;
        }
    }

    opterr = old_opterr;

    cout << "========================================" << endl
         << config.load_name << " Load Config" << endl
         << "========================================" << endl
         << "Base Threads: " << config.base_threads << endl
         << "Max Threads: " << config.max_threads << endl
         << "Test File: " << config.disk_path << endl
         << "File Size: " << config.file_size_mb << "MB" << endl
         << "Block Size: " << config.io_block_size_kb << "KB" << endl
         << "IO Mode: " << (config.sequential ? "Sequential" : "Random") << " | "
         << (config.read_only ? "Read-Only" : "Read-Write") << endl
         << "Fluctuation: " << config.load_fluctuation << "%" << endl
         << "Runtime: " << config.runtime_sec << "s" << endl
         << "Dynamic Adjust: " << (config.dynamic_adjust ? "Enabled" : "Disabled") << endl
         << "========================================" << endl;
}

int main(int argc, char* argv[]) {
    Config config;
    Stats stats;
    parse_args(argc, argv, config);

    // 创建测试文件
    size_t file_size = config.file_size_mb * 1024 * 1024;
    int fd = open(config.disk_path.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (fd < 0) {
        perror("Failed to open test file");
        return 1;
    }
    // 预分配文件大小
    if (ftruncate(fd, file_size) < 0) {
        perror("Failed to truncate file");
        close(fd);
        return 1;
    }
    cout << "Created test file: " << config.disk_path << " (" << config.file_size_mb << "MB)" << endl;

    // 启动负载控制器
    load_controller(config, stats, disk_io_task, ref(config), ref(stats), fd, file_size);

    // 清理测试文件
    close(fd);
    if (!config.read_only) {
        unlink(config.disk_path.c_str());
        cout << "Deleted test file: " << config.disk_path << endl;
    }

    // 最终报告
    double elapsed = config.runtime_sec;
    double total_gb = stats.io_bytes / 1e9;
    double avg_latency_us = stats.total_ops > 0 ? (double)stats.io_latency_us / stats.total_ops : 0;
    cout << "========================================" << endl
         << config.load_name << " Load Report" << endl
         << "========================================" << endl
         << "Total Runtime: " << elapsed << "s" << endl
         << "Average Threads: " << fixed << setprecision(1) << stats.current_threads << endl
         << "Total IO Ops: " << stats.total_ops << " (" << stats.total_ops / elapsed << " IOPS)" << endl
         << "Total IO Data: " << fixed << setprecision(2) << total_gb << "GB" << endl
         << "Average IO Bandwidth: " << fixed << setprecision(2) << (total_gb / elapsed) << "GB/s" << endl
         << "Average IO Latency: " << fixed << setprecision(1) << avg_latency_us << "us" << endl
         << "Average CPU Usage: " << fixed << setprecision(1)
         << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "%" << endl
         << "========================================" << endl;

    return 0;
}