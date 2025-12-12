#include "load_common.h"

// 网络IO负载：TCP客户端/服务器模式（此处实现客户端，需配合服务器使用）
struct Config : BaseConfig {
    string server_ip;       // 服务器IP
    int server_port;        // 服务器端口
    int pkt_size_bytes;     // 数据包大小（字节）
    int conn_per_thread;    // 每个线程的连接数
};

struct Stats : BaseStats {
    atomic<uint64_t> pkt_count = 0;  // 总数据包数
    atomic<uint64_t> net_bytes = 0;  // 总网络字节数
    atomic<uint64_t> net_latency_us = 0;  // 总网络延迟（微秒）
};

// 单个TCP连接的IO任务
void tcp_conn_task(int duration_ms, double load_factor, int sockfd, int pkt_size, Stats& stats) {
    vector<char> send_buf(pkt_size, 0);
    vector<char> recv_buf(pkt_size, 0);
    generate(send_buf.begin(), send_buf.end(), []() { return uniform_int_distribution<>(0, 255)(rng); });

    uint64_t local_pkts = 0;
    uint64_t local_bytes = 0;
    uint64_t local_latency_us = 0;

    auto start = high_resolution_clock::now();
    while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < duration_ms) {
        if (load_factor < 1.0 && uniform_real_distribution<>(0.0, 1.0)(rng) > load_factor) {
            this_thread::sleep_for(milliseconds(1));
            continue;
        }

        // 发送数据包并等待响应（echo模式）
        auto net_start = high_resolution_clock::now();
        ssize_t ret_send = send(sockfd, send_buf.data(), pkt_size, 0);
        if (ret_send != pkt_size) break;

        ssize_t ret_recv = recv(sockfd, recv_buf.data(), pkt_size, 0);
        auto net_end = high_resolution_clock::now();
        if (ret_recv != pkt_size) break;

        local_pkts++;
        local_bytes += pkt_size * 2;  // 发送+接收
        local_latency_us += duration_cast<microseconds>(net_end - net_start).count();
    }

    stats.pkt_count += local_pkts;
    stats.net_bytes += local_bytes;
    stats.net_latency_us += local_latency_us;
}

// 线程级网络IO任务（管理多个TCP连接）
void net_io_task(int duration_ms, double load_factor, Config& config, Stats& stats) {
    vector<int> sockfds;
    int pkt_size = config.pkt_size_bytes;

    // 建立多个TCP连接
    for (int i = 0; i < config.conn_per_thread; ++i) {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) continue;

        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(config.server_port);
        inet_pton(AF_INET, config.server_ip.c_str(), &server_addr.sin_addr);

        if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sockfd);
            continue;
        }
        sockfds.push_back(sockfd);
    }

    if (sockfds.empty()) {
        cerr << "Failed to establish any TCP connection" << endl;
        return;
    }

    // 并行处理多个连接
    vector<thread> conn_threads;
    for (int sockfd : sockfds) {
        conn_threads.emplace_back(tcp_conn_task, duration_ms, load_factor, sockfd, pkt_size, ref(stats));
    }

    for (auto& t : conn_threads) t.join();
    for (int sockfd : sockfds) close(sockfd);

    stats.total_ops += sockfds.size() * (duration_ms / 1000);
    atomic_double_add(stats.total_cpu_time, duration_ms / 1000.0);
}

void parse_args(int argc, char* argv[], Config& config) {
    parse_base_args(argc, argv, config);
    config.load_name = "Network-IO-Intensive";
    config.server_ip = "127.0.0.1";
    config.server_port = 8080;
    config.pkt_size_bytes = 1024;
    config.conn_per_thread = 2;

    const char* short_opts = "i:P:s:c:";
    const option long_opts[] = {
        {"ip", required_argument, nullptr, 'i'},
        {"port", required_argument, nullptr, 'P'},
        {"pkt-size", required_argument, nullptr, 's'},
        {"conn-per-thread", required_argument, nullptr, 'c'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'i': config.server_ip = optarg; break;
            case 'P': config.server_port = max(1024, stoi(optarg)); break;
            case 's': config.pkt_size_bytes = max(64, stoi(optarg)); break;
            case 'c': config.conn_per_thread = max(1, stoi(optarg)); break;
        }
    }

    cout << "========================================" << endl
         << config.load_name << " Load Config" << endl
         << "========================================" << endl
         << "Base Threads: " << config.base_threads << endl
         << "Max Threads: " << config.max_threads << endl
         << "Server: " << config.server_ip << ":" << config.server_port << endl
         << "Packet Size: " << config.pkt_size_bytes << "B" << endl
         << "Conn Per Thread: " << config.conn_per_thread << endl
         << "Fluctuation: " << config.load_fluctuation << "%" << endl
         << "Runtime: " << config.runtime_sec << "s" << endl
         << "Dynamic Adjust: " << (config.dynamic_adjust ? "Enabled" : "Disabled") << endl
         << "========================================" << endl;
}

// 简单TCP回显服务器（用于测试，可独立运行）
void echo_server(int port) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("Server socket creation failed");
        return;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Server bind failed");
        close(listen_fd);
        return;
    }

    listen(listen_fd, 1024);
    cout << "Echo server started on port " << port << endl;

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int conn_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
        if (conn_fd < 0) continue;

        thread([conn_fd]() {
            char buf[4096];
            while (true) {
                ssize_t ret = recv(conn_fd, buf, sizeof(buf), 0);
                if (ret <= 0) break;
                send(conn_fd, buf, ret, 0);
            }
            close(conn_fd);
        }).detach();
    }
}

int main(int argc, char* argv[]) {
    // 若带--server参数，启动回显服务器
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--server") {
            int port = (i+1 < argc) ? stoi(argv[i+1]) : 8080;
            echo_server(port);
            return 0;
        }
    }

    // 否则启动网络IO客户端负载
    Config config;
    Stats stats;
    parse_args(argc, argv, config);

    // 启动负载控制器
    load_controller(config, stats, net_io_task, ref(config), ref(stats));

    // 最终报告
    double elapsed = config.runtime_sec;
    double total_gb = stats.net_bytes / 1e9;
    double avg_latency_us = stats.pkt_count > 0 ? (double)stats.net_latency_us / stats.pkt_count : 0;
    cout << "========================================" << endl
         << config.load_name << " Load Report" << endl
         << "========================================" << endl
         << "Total Runtime: " << elapsed << "s" << endl
         << "Average Threads: " << fixed << setprecision(1) << stats.current_threads << endl
         << "Total Packets: " << stats.pkt_count / 1e6 << "M" << endl
         << "Total Network Data: " << fixed << setprecision(2) << total_gb << "GB" << endl
         << "Average Bandwidth: " << fixed << setprecision(2) << (total_gb / elapsed) << "GB/s" << endl
         << "Average Latency: " << fixed << setprecision(1) << avg_latency_us << "us" << endl
         << "Average CPU Usage: " << fixed << setprecision(1)
         << (stats.total_cpu_time / (elapsed * stats.current_threads)) * 100 << "%" << endl
         << "========================================" << endl;

    return 0;
}