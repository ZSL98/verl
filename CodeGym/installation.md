Run on 253

docker run --name agent-zsl --privileged=true --shm-size=10.24gb --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -v /mnt/nvme1/agent:/workspace/agent -v /mnt/nvme2:/workspace/external --net=host -it lmcache-ascend:v0.3.3-vllm-ascend-v0.9.2rc1-910b-cann-8.2rc1-py3.11-openeuler-22.03 bash

dnf -y install podman

# The following is required if you want to run podman inside of a container
cp /usr/share/containers/containers.conf /etc/containers/containers.conf
sed -i 's/^#cgroup_manager = "systemd"/cgroup_manager = "cgroupfs"/' /etc/containers/containers.conf
sed -i 's/^#events_logger = "journald"/events_logger = "file"/' /etc/containers/containers.conf

<!-- # add below in /etc/containers/containers.conf
[containers]
netns="host"
userns="host"
ipcns="host"
utsns="host"
cgroupns="host"
pidns="host"
cgroups="enabled"
log_driver = "k8s-file"
[engine]
runtime="crun" -->

cp /usr/share/containers/storage.conf /etc/containers/storage.conf
sed -i 's/^#ignore_chown_errors = "false"/ignore_chown_errors = "true"/' /etc/containers/storage.conf

# uncomment below in /etc/containers/storage.conf
mount_program = "/usr/bin/fuse-overlayfs"

# Try below
podman pull docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-24661:latest

# Run below
uv run --extra full mini-extra swebench --model /workspace/external/Qwen3-32B --subset verified --split test --workers 8 --output ./output/