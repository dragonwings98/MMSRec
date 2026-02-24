#!/bin/bash
# 图束模型超参调优一键运行脚本
# 使用说明：
# 1. 测试模式（快速验证）：./run_bundle.sh test
# 2. 正式调优（完整100组配置）：./run_bundle.sh
# 3. 仅训练最优配置：./run_bundle.sh train

# 设置环境变量（避免CUDA相关问题）
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/li/projects/MMSRec"

# 配置文件路径
CONFIG_PATH="config/config.yaml"

# 日志目录（自动创建）
LOG_DIR="experiment_results/logs"
mkdir -p ${LOG_DIR}

# 根据参数执行不同逻辑
if [ "$1" = "test" ]; then
    # 测试模式：短训练+快速验证
    echo "===== Start the beam model test mode ====="
    python test_short_train.py > ${LOG_DIR}/test_mode.log 2>&1
    echo "✅ Test mode complete！log：${LOG_DIR}/test_mode.log"

elif [ "$1" = "train" ]; then
    # 仅训练最优配置（调优完成后使用）
    echo "===== Start optimal configuration and complete training ====="
    python train/train_bundle.py --config ${CONFIG_PATH} > ${LOG_DIR}/best_train.log 2>&1
    echo "✅ Optimal configuration training completed！日志：${LOG_DIR}/best_train.log"

else
    # 正式调优模式：完整超参搜索+可视化
    echo "===== 启动图束模型超参调优（约100组配置）====="
    # 后台运行（即使断开SSH也能继续），并保存日志
    nohup python train/hyperparam_tuning_bundle.py --config ${CONFIG_PATH} > ${LOG_DIR}/tuning.log 2>&1 &
    echo "✅ 超参调优已后台启动！"
    echo "📄 日志文件：${LOG_DIR}/tuning.log"
    echo "🔍 查看进度：tail -f ${LOG_DIR}/tuning.log"
fi
