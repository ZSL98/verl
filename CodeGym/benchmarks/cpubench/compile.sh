#!/bin/bash
# 编译所有负载程序
g++ -std=c++17 -O3 -pthread compute_intensive.cpp -o compute_intensive -lm
g++ -std=c++17 -O3 -pthread mem_intensive.cpp -o mem_intensive -lm
g++ -std=c++17 -O3 -pthread cache_sensitive.cpp -o cache_sensitive -lm
g++ -std=c++17 -O3 -pthread io_disk_intensive.cpp -o io_disk_intensive -lm
g++ -std=c++17 -O3 -pthread io_network_intensive.cpp -o io_network_intensive -lm
g++ -std=c++17 -O3 -pthread short_batch_task.cpp -o short_batch_task -lm

echo "All load programs compiled successfully!"