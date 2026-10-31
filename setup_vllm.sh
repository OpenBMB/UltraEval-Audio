#!/bin/bash
# set -e  # 遇到错误立即退出
set -x  # 打印执行日志

# ================= 配置区域 =================
# 修正路径获取方式
ALGORITHM_PATH=$(pwd)
# 定义服务端口
PORT=8630
# 定义服务名称
SERVED_MODEL_NAME="qwen3omni"


VLLM_MODEL_PATH="/opt/huawei/dataset/data/modelpt/Qwen_omni/Qwen3-Omni-30B-A3B-Instruct"



echo ">>> [Init] Displaying the NPU memory...."
npu-smi info


# ================= 环境准备 =================
echo ">>> [Init] Checking environment..."
#cd /opt/huawei/dataset/Audio_dataset/framework/ms-swift-main/
#pip install -e .
pip install --upgrade pip
pip install vllm==0.13.0
pip install vllm-ascend==0.13.0rc1
pip install transformers==4.57.1
pip install torchvision torchaudio
pip install accelerate==1.10.1
pip install deepspeed 
pip install qwen_omni_utils
pip install msgspec
pip install urllib3==1.26.0 numpy==1.26.4 requests
pip install "decord" -U

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

pip list | grep transformers
pip list | grep torch
pip list | grep vllm

# ================= 环境变量设置 (Ascend NPU) =================
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ================= 启动 vLLM 后端 =================
echo ">>> [Start] Starting vLLM server in background..."

export USE_VLLM=1

# 1. 【核心修复】定义正确的日志目录和文件名
LOG_DIR="${ALGORITHM_PATH}/log"
mkdir -p "${LOG_DIR}"  # 确保目录存在

# 使用 SERVED_MODEL_NAME (qwen3vl) 作为文件名，不再使用未定义的 benchmarklist
SERVER_LOG="${LOG_DIR}/vllm_server_${SERVED_MODEL_NAME}.log"
SERVER_JUDGE_LOG="${LOG_DIR}/vllm_judge_server_${SERVED_MODEL_NAME}.log"

echo ">>> [Log] Log file path: ${SERVER_LOG}"

touch "${SERVER_LOG}"

# ================= 启动 vLLM 后端 =================
echo ">>> [Start] Starting vLLM server in background..."

# 1. 启动 vLLM，日志依然写入文件 (这样为了保证 SERVER_PID 能抓到正确的 vLLM 进程)
ASCEND_RT_VISIBLE_DEVICES=0,1 \
nohup vllm serve ${VLLM_MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 2 \
    --trust-remote-code \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.98 > ${SERVER_LOG} 2>&1 &

# 获取 vLLM 进程 ID
SERVER_PID=$!
echo ">>> [Start] vLLM server started with PID: ${SERVER_PID}"
echo ">>> [Log] Log file is located at: ${SERVER_LOG}"

# 2. 【核心修改】启动 tail 在后台实时打印日志到屏幕
echo ">>> [Log] Streaming logs to console..."
tail -f ${SERVER_LOG} &
TAIL_PID=$!  # 记录 tail 的 PID，以便稍后关闭

# 3. 修改清理函数：退出时同时杀掉 vLLM 和 tail 进程
cleanup() {
    echo ">>> [Exit] Cleaning up..."
    kill ${TAIL_PID} 2>/dev/null || true  # 先停止打印日志
    kill ${SERVER_PID} 2>/dev/null || true # 再停止服务
}
trap cleanup EXIT

# ================= 等待服务就绪 =================
echo ">>> [Wait] Waiting for vLLM to be ready on port ${PORT}..."

MAX_RETRIES=600  # 最多等待 60 * 5 = 300秒
for ((i=1; i<=MAX_RETRIES; i++)); do
    # 检查 /health 或 /v1/models 接口
    if curl -s http://localhost:${PORT}/v1/models > /dev/null; then
        echo ">>> [Wait] vLLM server is READY!"
        break
    fi
    echo ">>> [Wait] Server not ready yet... (Attempt $i/$MAX_RETRIES). Sleeping 5s..."
    sleep 5
done


if ! curl -s http://localhost:${PORT}/v1/models > /dev/null; then
    echo ">>> [Error] Server failed to start. Check logs below:"
    tail -n 50 ${SERVER_LOG}
    exit 1
fi


# ================= 执行测试 =================
echo ">>> [Test] Running inference test..."
export TEST_IMAGE_PATH="${ALGORITHM_PATH}/test.jpg"
export SERVED_MODEL_NAME="qwen3omni"
export VLLM_PORT=8630

python3 run_test.py

echo ">>> [Done] Script finished successfully." 


# 检查所有进程是否成功执行
if [ $? -eq 0 ]; then
    echo "所有部署进程成功启动！"
else
    echo "某些部署进程失败。"
fi
