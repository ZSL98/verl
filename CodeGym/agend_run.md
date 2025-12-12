77节点部署agent：

容器启动：

docker run --name agent-zsl-2 --privileged=true --shm-size=10.24gb --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.inf -v /home/model:/workspace/model -v /home/zsl/code_repo/agent:/workspace/agent  --net=host -it quay.io/ascend/vllm-ascend:latest bash

容器内terminal1：
`ASCEND_RT_VISIBLE_DEVICES=2,3 vllm serve /workspace/model/Qwen3-8B --served-model-name Qwen3-8B --tensor-parallel-size 2 --max-model-len 40960 --host 0.0.0.0 --port 8008 --enable-auto-tool-choice --tool-call-parser qwen3_coder`

容器内terminal2, under /workspace/agent/mini-swe-agent/:
`export MSWEA_COST_TRACKING='ignore_errors'`
`uv run mini-extra config set HOSTED_VLLM_API_BASE "http://0.0.0.0:8008/v1"`
`uv run mini-extra swebench-single --model hosted_vllm/Qwen3-8B --subset pcmoritz/cpython_dataset --split train --instance python__cpython-137669 --environment-class bubblewrap --output ../data/python__cpython-137669.traj.json`


首先安装bubblewrap，先保证yum在容器内可用
`vim /etc/yum.repos.d/openEuler.repo`
```bash
[openEuler-everything]
name=openEuler-everything
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-24.03-LTS-SP2/everything/aarch64/
enabled=1
gpgcheck=0
gpgkey=http://mirrors.tools.huawei.com/openeuler/openEuler-24.03-LTS-SP2/everything/aarch64/RPM-GPG-KEY-openEuler

[openEuler-EPOL]
name=openEuler-epol
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-24.03-LTS-SP2/EPOL/main/aarch64/
enabled=1
gpgcheck=0

[openEuler-update]
name=openEuler-update
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-24.03-LTS-SP2/update/aarch64/
enabled=1
gpgcheck=0
```
`yum install -y bubblewrap`
`pip install uv`
`uv add datasets`