#!/bin/bash
set -euo pipefail

# ========================================
# 基础配置（适配Kunpeng-920硬件）
# ========================================
TEST_DURATION=10          # 每个测试运行时间（秒）
OUTPUT_DIR="./numa_bind_mix_results"
mkdir -p $OUTPUT_DIR

# 绑核策略配置（按NUMA节点隔离）
# 计算密集型：NUMA 0-1（CPU 0-47），48线程
COMPUTE_NUMA_NODES="0-1"
COMPUTE_CPUS="0-47"
COMPUTE_THREADS=48

# 访存密集型：NUMA 2-3（CPU 48-95），48线程
MEM_NUMA_NODES="2-3"
MEM_CPUS="48-95"
MEM_THREADS=48
MEM_SIZE_MB=4096          # 内存缓冲区4GB（适配NUMA节点本地内存）
MEM_GRANULARITY=64
MEM_SEQUENTIAL=0

# 缓存敏感型：NUMA 4-5（CPU 96-143），48线程
CACHE_NUMA_NODES="4-5"
CACHE_CPUS="96-143"
CACHE_THREADS=48
CACHE_SIZE_MB=64          # 缓存占用64MB（小于单个NUMA节点L3缓存24MB×2=48MB？调整为40MB）
CACHE_SIZE_MB=40

# 磁盘IO密集型：NUMA 6-7（CPU 144-191），48线程
DISK_NUMA_NODES="6-7"
DISK_CPUS="144-191"
DISK_THREADS=48
DISK_TEST_FILE="./disk_test.tmp"
DISK_FILE_SIZE_MB=1024    # 测试文件4GB
DISK_BLOCK_SIZE_KB=8     # 块大小64KB（优化磁盘IO效率）
DISK_SEQUENTIAL=0
DISK_READ_ONLY=0

# 工具依赖检查
check_dependency() {
    if ! command -v numactl &> /dev/null; then
        echo "错误：未安装numactl（NUMA绑核工具），请执行：sudo yum install numactl 或 sudo apt install numactl"
        exit 1
    fi
    if ! command -v bc &> /dev/null; then
        echo "错误：未安装bc（浮点数计算工具），请执行：sudo yum install bc 或 sudo apt install bc"
        exit 1
    fi
}

# ========================================
# 工具函数
# ========================================
# 1. 记录性能数据
record_perf_data() {
    local load_name=$1
    local run_mode=$2  # single:单独运行, mix_no_bind:无绑核混部, mix_bind:绑核混部
    local log_file=$3
    local perf_file="$OUTPUT_DIR/${load_name}_${run_mode}_perf.txt"

    case $load_name in
        "compute")
            total_float_ops=$(grep "Total Float Ops" $log_file | awk '{print $4}')
            throughput=$(grep "Average Throughput" $log_file | awk '{print $3}')
            cpu_usage=$(grep "Average CPU Usage" $log_file | awk '{print $4}' | sed 's/%//')
            echo "total_float_ops_B=$total_float_ops" > $perf_file
            echo "throughput_Mops_s=$throughput" >> $perf_file
            echo "cpu_usage_pct=$cpu_usage" >> $perf_file
            ;;
        "mem")
            total_mem_gb=$(grep "Total Mem Access" $log_file | awk '{print $4}')
            bandwidth=$(grep "Average Bandwidth" $log_file | awk '{print $3}')
            cache_misses=$(grep "Estimated Cache Misses" $log_file | awk '{print $4}')
            cpu_usage=$(grep "Average CPU Usage" $log_file | awk '{print $4}' | sed 's/%//')
            echo "total_mem_gb=$total_mem_gb" > $perf_file
            echo "bandwidth_GB_s=$bandwidth" >> $perf_file
            echo "cache_misses_M=$cache_misses" >> $perf_file
            echo "cpu_usage_pct=$cpu_usage" >> $perf_file
            ;;
        "cache")
            hit_rate=$(grep "Cache Hit Rate" $log_file | awk '{print $4}' | sed 's/%//')
            total_ops=$(grep "Total Ops" $log_file | awk '{print $3}')
            cpu_usage=$(grep "Average CPU Usage" $log_file | awk '{print $4}' | sed 's/%//')
            echo "cache_hit_rate_pct=$hit_rate" > $perf_file
            echo "total_ops_M=$total_ops" >> $perf_file
            echo "cpu_usage_pct=$cpu_usage" >> $perf_file
            ;;
        "disk")
            iops=$(grep "Total IO Ops" $log_file | awk '{print $4 " " $5}' | sed 's/(//;s/)//')
            bandwidth=$(grep "Average IO Bandwidth" $log_file | awk '{print $4}')
            latency=$(grep "Average IO Latency" $log_file | awk '{print $4}')
            cpu_usage=$(grep "Average CPU Usage" $log_file | awk '{print $4}' | sed 's/%//')
            echo "iops=$iops" > $perf_file
            echo "bandwidth_GB_s=$bandwidth" >> $perf_file
            echo "avg_latency_us=$latency" >> $perf_file
            echo "cpu_usage_pct=$cpu_usage" >> $perf_file
            ;;
    esac
    echo "✅ 测试完成：$load_name ($run_mode)，性能数据保存至 $perf_file"
}

# 2. 计算性能变化率
calculate_change() {
    local base_val=$1
    local test_val=$2
    change_rate=$(echo "scale=2; (($test_val - $base_val) / $base_val) * 100" | bc)
    echo $change_rate
}

# 3. 运行单个负载（支持绑核）
run_single_load() {
    local load_name=$1
    local bind=$2  # 0:不绑核, 1:绑核
    local log_file=$3

    echo -e "\n📌 运行 $load_name（$( [ $bind -eq 1 ] && echo "绑核" || echo "不绑核" )）..."
    case $load_name in
        "compute")
            if [ $bind -eq 1 ]; then
                numactl --cpunodebind=$COMPUTE_NUMA_NODES --membind=$COMPUTE_NUMA_NODES \
                    ./compute_intensive \
                    -t $COMPUTE_THREADS -T $COMPUTE_THREADS \
                    -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            else
                ./compute_intensive \
                    -t $COMPUTE_THREADS -T $COMPUTE_THREADS \
                    -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            fi
            ;;
        "mem")
            if [ $bind -eq 1 ]; then
                numactl --cpunodebind=$MEM_NUMA_NODES --membind=$MEM_NUMA_NODES \
                    ./mem_intensive \
                    -t $MEM_THREADS -T $MEM_THREADS \
                    -M $MEM_SIZE_MB -g $MEM_GRANULARITY -s $MEM_SEQUENTIAL \
                    -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            else
                ./mem_intensive \
                    -t $MEM_THREADS -T $MEM_THREADS \
                    -M $MEM_SIZE_MB -g $MEM_GRANULARITY -s $MEM_SEQUENTIAL \
                    -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            fi
            ;;
        "cache")
            if [ $bind -eq 1 ]; then
                numactl --cpunodebind=$CACHE_NUMA_NODES --membind=$CACHE_NUMA_NODES \
                    ./cache_sensitive \
                    -t $CACHE_THREADS -T $CACHE_THREADS \
                    -C $CACHE_SIZE_MB -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            else
                ./cache_sensitive \
                    -t $CACHE_THREADS -T $CACHE_THREADS \
                    -C $CACHE_SIZE_MB -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            fi
            ;;
        "disk")
            if [ $bind -eq 1 ]; then
                numactl --cpunodebind=$DISK_NUMA_NODES --membind=$DISK_NUMA_NODES \
                    ./io_disk_intensive \
                    -t $DISK_THREADS -T $DISK_THREADS \
                    -p $DISK_TEST_FILE -F $DISK_FILE_SIZE_MB -b $DISK_BLOCK_SIZE_KB \
                    -s $DISK_SEQUENTIAL -R $DISK_READ_ONLY \
                    -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            else
                ./io_disk_intensive \
                    -t $DISK_THREADS -T $DISK_THREADS \
                    -p $DISK_TEST_FILE -F $DISK_FILE_SIZE_MB -b $DISK_BLOCK_SIZE_KB \
                    -s $DISK_SEQUENTIAL -R $DISK_READ_ONLY \
                    -f 0 -d 0 -r $TEST_DURATION \
                    > $log_file 2>&1
            fi
            # # 清理磁盘测试文件
            # [ -f $DISK_TEST_FILE ] && rm -f $DISK_TEST_FILE
            ;;
    esac
}

# ========================================
# 测试流程
# ========================================
check_dependency

# 第一步：单独运行各负载（基准测试，不绑核，获取无干扰性能）
echo "========================================"
echo "📊 第一步：单独运行基准测试（不绑核）"
echo "========================================"
run_single_load "compute" 0 "$OUTPUT_DIR/compute_single.log"
record_perf_data "compute" "single" "$OUTPUT_DIR/compute_single.log"

run_single_load "mem" 0 "$OUTPUT_DIR/mem_single.log"
record_perf_data "mem" "single" "$OUTPUT_DIR/mem_single.log"

run_single_load "cache" 0 "$OUTPUT_DIR/cache_single.log"
record_perf_data "cache" "single" "$OUTPUT_DIR/cache_single.log"

run_single_load "disk" 0 "$OUTPUT_DIR/disk_single.log"
record_perf_data "disk" "single" "$OUTPUT_DIR/disk_single.log"

# 第二步：无绑核混部测试
echo -e "\n========================================"
echo "📊 第二步：无绑核混部测试"
echo "========================================"
mix_no_bind_log_dir="$OUTPUT_DIR/mix_no_bind_logs"
mkdir -p $mix_no_bind_log_dir

# 后台运行所有负载（不绑核）
echo "🚀 启动无绑核混部（后台运行）..."
./compute_intensive \
    -t $COMPUTE_THREADS -T $COMPUTE_THREADS \
    -f 0 -d 0 -r $TEST_DURATION \
    > $mix_no_bind_log_dir/compute.log 2>&1 &
COMPUTE_PID=$!

./mem_intensive \
    -t $MEM_THREADS -T $MEM_THREADS \
    -M $MEM_SIZE_MB -g $MEM_GRANULARITY -s $MEM_SEQUENTIAL \
    -f 0 -d 0 -r $TEST_DURATION \
    > $mix_no_bind_log_dir/mem.log 2>&1 &
MEM_PID=$!

./cache_sensitive \
    -t $CACHE_THREADS -T $CACHE_THREADS \
    -C $CACHE_SIZE_MB -f 0 -d 0 -r $TEST_DURATION \
    > $mix_no_bind_log_dir/cache.log 2>&1 &
CACHE_PID=$!

./io_disk_intensive \
    -t $DISK_THREADS -T $DISK_THREADS \
    -p $DISK_TEST_FILE -F $DISK_FILE_SIZE_MB -b $DISK_BLOCK_SIZE_KB \
    -s $DISK_SEQUENTIAL -r $DISK_READ_ONLY \
    -f 0 -d 0 -r $TEST_DURATION \
    > $mix_no_bind_log_dir/disk.log 2>&1 &
DISK_PID=$!

# 等待完成
echo "⌛ 等待无绑核混部运行 $TEST_DURATION 秒..."
wait $COMPUTE_PID $MEM_PID $CACHE_PID $DISK_PID 2>/dev/null
[ -f $DISK_TEST_FILE ] && rm -f $DISK_TEST_FILE

# 记录无绑核混部性能
record_perf_data "compute" "mix_no_bind" "$mix_no_bind_log_dir/compute.log"
record_perf_data "mem" "mix_no_bind" "$mix_no_bind_log_dir/mem.log"
record_perf_data "cache" "mix_no_bind" "$mix_no_bind_log_dir/cache.log"
record_perf_data "disk" "mix_no_bind" "$mix_no_bind_log_dir/disk.log"

# 第三步：绑核混部测试（NUMA节点隔离）
echo -e "\n========================================"
echo "📊 第三步：绑核混部测试（NUMA节点隔离）"
echo "========================================"
mix_bind_log_dir="$OUTPUT_DIR/mix_bind_logs"
mkdir -p $mix_bind_log_dir

# 后台运行所有负载（绑核到指定NUMA节点）
echo "🚀 启动绑核混部（后台运行）..."
numactl --cpunodebind=$COMPUTE_NUMA_NODES --membind=$COMPUTE_NUMA_NODES \
    ./compute_intensive \
    -t $COMPUTE_THREADS -T $COMPUTE_THREADS \
    -f 0 -d 0 -r $TEST_DURATION \
    > $mix_bind_log_dir/compute.log 2>&1 &
COMPUTE_PID=$!

numactl --cpunodebind=$MEM_NUMA_NODES --membind=$MEM_NUMA_NODES \
    ./mem_intensive \
    -t $MEM_THREADS -T $MEM_THREADS \
    -M $MEM_SIZE_MB -g $MEM_GRANULARITY -s $MEM_SEQUENTIAL \
    -f 0 -d 0 -r $TEST_DURATION \
    > $mix_bind_log_dir/mem.log 2>&1 &
MEM_PID=$!

numactl --cpunodebind=$CACHE_NUMA_NODES --membind=$CACHE_NUMA_NODES \
    ./cache_sensitive \
    -t $CACHE_THREADS -T $CACHE_THREADS \
    -C $CACHE_SIZE_MB -f 0 -d 0 -r $TEST_DURATION \
    > $mix_bind_log_dir/cache.log 2>&1 &
CACHE_PID=$!

numactl --cpunodebind=$DISK_NUMA_NODES --membind=$DISK_NUMA_NODES \
    ./io_disk_intensive \
    -t $DISK_THREADS -T $DISK_THREADS \
    -p $DISK_TEST_FILE -F $DISK_FILE_SIZE_MB -b $DISK_BLOCK_SIZE_KB \
    -s $DISK_SEQUENTIAL -r $DISK_READ_ONLY \
    -f 0 -d 0 -r $TEST_DURATION \
    > $mix_bind_log_dir/disk.log 2>&1 &
DISK_PID=$!

# 等待完成
echo "⌛ 等待绑核混部运行 $TEST_DURATION 秒..."
wait $COMPUTE_PID $MEM_PID $CACHE_PID $DISK_PID 2>/dev/null
[ -f $DISK_TEST_FILE ] && rm -f $DISK_TEST_FILE

# 记录绑核混部性能
record_perf_data "compute" "mix_bind" "$mix_bind_log_dir/compute.log"
record_perf_data "mem" "mix_bind" "$mix_bind_log_dir/mem.log"
record_perf_data "cache" "mix_bind" "$mix_bind_log_dir/cache.log"
record_perf_data "disk" "mix_bind" "$mix_bind_log_dir/disk.log"

# ========================================
# 生成对比报告
# ========================================
echo -e "\n========================================"
echo "📋 绑核 vs 无绑核混部性能对比报告"
echo "========================================"
report_file="$OUTPUT_DIR/numa_bind_comparison_report.txt"

# 报告表头
cat > $report_file << EOF
# 绑核策略 vs 无绑核混部性能对比报告
# 硬件环境：Kunpeng-920（192核、8 NUMA节点、192MB L3缓存）
# 绑核策略：NUMA节点隔离（计算[0-1]、访存[2-3]、缓存[4-5]、磁盘IO[6-7]）
# 测试配置：每个负载48线程，运行$TEST_DURATION 秒，负载稳定（无波动、无动态调整）
# 测试时间：$(date)
# =======================================

EOF

# 加载所有性能数据
load_perf_data() {
    local load_name=$1
    echo "=== 加载 $load_name 性能数据 ===" >> $report_file
    # 单独运行数据
    single=$(cat "$OUTPUT_DIR/${load_name}_single_perf.txt")
    # 无绑核混部数据
    no_bind=$(cat "$OUTPUT_DIR/${load_name}_mix_no_bind_perf.txt")
    # 绑核混部数据
    bind=$(cat "$OUTPUT_DIR/${load_name}_mix_bind_perf.txt")
    echo "$single" >> $report_file
    echo "$no_bind" >> $report_file
    echo "$bind" >> $report_file
    echo "" >> $report_file
}

# 计算负载对比数据
generate_load_report() {
    local load_name=$1
    local core_metric=$2  # 核心对比指标（如throughput_Mops_s、bandwidth_GB_s）
    local metric_name=$3  # 指标名称（如吞吐量、内存带宽）
    local unit=$4         # 单位（如Mops/s、GB/s）

    echo "=== $load_name 负载对比 ===" >> $report_file

    # 提取核心指标
    single_val=$(grep "$core_metric" "$OUTPUT_DIR/${load_name}_single_perf.txt" | cut -d'=' -f2 | sed -E 's/[^0-9.]//g')
    no_bind_val=$(grep "$core_metric" "$OUTPUT_DIR/${load_name}_mix_no_bind_perf.txt" | cut -d'=' -f2 | sed -E 's/[^0-9.]//g')
    bind_val=$(grep "$core_metric" "$OUTPUT_DIR/${load_name}_mix_bind_perf.txt" | cut -d'=' -f2 | sed -E 's/[^0-9.]//g')

    # 计算变化率
    no_bind_change=$(calculate_change $single_val $no_bind_val)
    bind_change=$(calculate_change $single_val $bind_val)
    bind_vs_no_bind=$(calculate_change $no_bind_val $bind_val)

    # 写入报告
    cat >> $report_file << EOF
- $metric_name（$unit）：
  单独运行：$single_val
  无绑核混部：$no_bind_val（变化：$no_bind_change%）
  绑核混部：$bind_val（变化：$bind_change%）
  绑核vs无绑核：性能提升 $bind_vs_no_bind%

EOF

    # 补充其他关键指标（根据负载类型）
    case $load_name in
        "compute")
            single_cpu=$(grep "cpu_usage_pct" "$OUTPUT_DIR/${load_name}_single_perf.txt" | cut -d'=' -f2)
            no_bind_cpu=$(grep "cpu_usage_pct" "$OUTPUT_DIR/${load_name}_mix_no_bind_perf.txt" | cut -d'=' -f2)
            bind_cpu=$(grep "cpu_usage_pct" "$OUTPUT_DIR/${load_name}_mix_bind_perf.txt" | cut -d'=' -f2)
            echo "- CPU使用率（%）：" >> $report_file
            echo "  单独运行：$single_cpu → 无绑核混部：$no_bind_cpu → 绑核混部：$bind_cpu" >> $report_file
            ;;
        "mem")
            single_cache_miss=$(grep "cache_misses_M" "$OUTPUT_DIR/${load_name}_single_perf.txt" | cut -d'=' -f2)
            no_bind_cache_miss=$(grep "cache_misses_M" "$OUTPUT_DIR/${load_name}_mix_no_bind_perf.txt" | cut -d'=' -f2)
            bind_cache_miss=$(grep "cache_misses_M" "$OUTPUT_DIR/${load_name}_mix_bind_perf.txt" | cut -d'=' -f2)
            echo "- 缓存缺失数（M）：" >> $report_file
            echo "  单独运行：$single_cache_miss → 无绑核混部：$no_bind_cache_miss → 绑核混部：$bind_cache_miss" >> $report_file
            ;;
        "cache")
            single_hit_rate=$(grep "cache_hit_rate_pct" "$OUTPUT_DIR/${load_name}_single_perf.txt" | cut -d'=' -f2)
            no_bind_hit_rate=$(grep "cache_hit_rate_pct" "$OUTPUT_DIR/${load_name}_mix_no_bind_perf.txt" | cut -d'=' -f2)
            bind_hit_rate=$(grep "cache_hit_rate_pct" "$OUTPUT_DIR/${load_name}_mix_bind_perf.txt" | cut -d'=' -f2)
            echo "- 缓存命中率（%）：" >> $report_file
            echo "  单独运行：$single_hit_rate → 无绑核混部：$no_bind_hit_rate → 绑核混部：$bind_hit_rate" >> $report_file
            ;;
        "disk")
            single_latency=$(grep "avg_latency_us" "$OUTPUT_DIR/${load_name}_single_perf.txt" | cut -d'=' -f2)
            no_bind_latency=$(grep "avg_latency_us" "$OUTPUT_DIR/${load_name}_mix_no_bind_perf.txt" | cut -d'=' -f2)
            bind_latency=$(grep "avg_latency_us" "$OUTPUT_DIR/${load_name}_mix_bind_perf.txt" | cut -d'=' -f2)
            echo "- IO延迟（us）：" >> $report_file
            echo "  单独运行：$single_latency → 无绑核混部：$no_bind_latency → 绑核混部：$bind_latency" >> $report_file
            ;;
    esac
    echo "" >> $report_file
}

# 生成各负载报告
generate_load_report "compute" "throughput_Mops_s" "吞吐量" "Mops/s"
generate_load_report "mem" "bandwidth_GB_s" "内存带宽" "GB/s"
generate_load_report "cache" "total_ops_M" "总操作数" "M"
generate_load_report "disk" "iops" "IOPS" ""


# 打印报告到控制台
cat $report_file
echo -e "\n🎉 测试完成！报告已保存至：$report_file"
echo -e "所有日志文件保存至：$OUTPUT_DIR"