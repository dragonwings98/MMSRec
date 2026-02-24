#!/bin/bash
# 图束模型全自动分层超参调优脚本
# 使用说明：./auto_tuning.sh [配置文件路径]
# 示例：./auto_tuning.sh config/config.yaml

# ===================== 配置项 =====================
# 默认配置文件路径
CONFIG_PATH=${1:-"config/config.yaml"}
# 结果保存目录
RESULT_DIR="experiment_results"
# Python脚本路径
TUNING_SCRIPT="train/hyperparam_tuning_bundle.py"
# 日志目录
LOG_DIR="${RESULT_DIR}/auto_tuning_logs"

# ===================== 工具函数 =====================
# 读取最优配置函数（从txt文件中提取指定参数）
get_best_param() {
    local stage=$1
    local param=$2
    local best_file="${RESULT_DIR}/best_config_${stage}.txt"
    # 提取参数值（兼容数值/字符串）
    grep "^${param}:" ${best_file} | awk -F': ' '{print $2}' | tr -d '"'
}

# 检查脚本执行状态
check_status() {
    if [ $? -ne 0 ]; then
        echo "❌ error: $1 stage failed！"
        exit 1
    fi
}

# ===================== 初始化 =====================
echo "===== Fully Automated Layered Hyperparameter Tuning of the Bundle Model ====="
echo "start_time：$(date +%Y-%m-%d\ %H:%M:%S)"
echo "config：${CONFIG_PATH}"
echo "result_dir：${RESULT_DIR}"

# 创建目录
mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

# 检查必要文件
if [ ! -f ${CONFIG_PATH} ]; then
    echo "❌ error：config ${CONFIG_PATH} not exist！"
    exit 1
fi

if [ ! -f ${TUNING_SCRIPT} ]; then
    echo "❌ error：sh ${TUNING_SCRIPT} not exist！"
    exit 1
fi

# ===================== 阶段1：核心层调优 =====================
echo -e "\n===== Start core stage tuning.====="
echo "goal：lr、node_stalk_dim、fusion_type、batch_size"
echo "log：${LOG_DIR}/core_tuning.log"

# 运行核心层调优（后台运行+日志保存）
nohup python ${TUNING_SCRIPT} \
    --config ${CONFIG_PATH} \
    --stage core > ${LOG_DIR}/core_tuning.log 2>&1

# 等待执行完成
wait
check_status "core"

# 读取核心层最优配置
echo -e "\n===== Extract the optimal configuration of the core ====="
BEST_LR=$(get_best_param "core" "lr")
BEST_NODE_DIM=$(get_best_param "core" "node_stalk_dim")
BEST_FUSION=$(get_best_param "core" "fusion_type")
BEST_BATCH_SIZE=$(get_best_param "core" "batch_size")

# 验证提取结果
if [ -z "${BEST_LR}" ] || [ -z "${BEST_NODE_DIM}" ] || [ -z "${BEST_FUSION}" ] || [ -z "${BEST_BATCH_SIZE}" ]; then
    echo "❌ Error: Failed to extract optimal configuration for core stage！"
    exit 1
fi

echo "best config of core："
echo "  lr: ${BEST_LR}"
echo "  node_stalk_dim: ${BEST_NODE_DIM}"
echo "  fusion_type: ${BEST_FUSION}"
echo "  batch_size: ${BEST_BATCH_SIZE}"

# ===================== 阶段2：次要层调优 =====================
echo -e "\n===== Start secondary stage tuning ====="
echo "goal：edge_stalk_dim、num_diffusion_layers、weight_decay、dropout_rate"
echo "based on ：lr=${BEST_LR}, node_dim=${BEST_NODE_DIM}, fusion=${BEST_FUSION}, bs=${BEST_BATCH_SIZE}"
echo "log：${LOG_DIR}/secondary_tuning.log"

# 运行次要层调优（传入核心层最优配置）
nohup python ${TUNING_SCRIPT} \
    --config ${CONFIG_PATH} \
    --stage secondary \
    --best_lr ${BEST_LR} \
    --best_node_dim ${BEST_NODE_DIM} \
    --best_fusion ${BEST_FUSION} \
    --best_batch_size ${BEST_BATCH_SIZE} > ${LOG_DIR}/secondary_tuning.log 2>&1

# 等待执行完成
wait
check_status "Secondary"

# 读取次要层最优配置
echo -e "\n===== Extracting the optimal configuration of the secondary stage ====="
BEST_EDGE_DIM=$(get_best_param "secondary" "edge_stalk_dim")
BEST_LAYERS=$(get_best_param "secondary" "num_diffusion_layers")
BEST_WD=$(get_best_param "secondary" "weight_decay")
BEST_DROPOUT=$(get_best_param "secondary" "dropout_rate")

# 验证提取结果
if [ -z "${BEST_EDGE_DIM}" ] || [ -z "${BEST_LAYERS}" ] || [ -z "${BEST_WD}" ] || [ -z "${BEST_DROPOUT}" ]; then
    echo "❌ error：Secondary optimal configuration extraction failed.！"
    exit 1
fi

echo "Optimal configuration of secondary："
echo "  edge_stalk_dim: ${BEST_EDGE_DIM}"
echo "  num_diffusion_layers: ${BEST_LAYERS}"
echo "  weight_decay: ${BEST_WD}"
echo "  dropout_rate: ${BEST_DROPOUT}"

# ===================== 阶段3：微调层调优 =====================
echo -e "\n===== Start fine-tuning optimization====="
echo "goal: Fine-tune the nearest values of core parameters"
echo "log：${LOG_DIR}/fine_tuning.log"

# 运行微调层调优（传入核心+次要层最优配置）
nohup python ${TUNING_SCRIPT} \
    --config ${CONFIG_PATH} \
    --stage fine \
    --best_lr ${BEST_LR} \
    --best_node_dim ${BEST_NODE_DIM} \
    --best_fusion ${BEST_FUSION} \
    --best_batch_size ${BEST_BATCH_SIZE} \
    --best_edge_dim ${BEST_EDGE_DIM} \
    --best_layers ${BEST_LAYERS} \
    --best_wd ${BEST_WD} \
    --best_dropout ${BEST_DROPOUT} > ${LOG_DIR}/fine_tuning.log 2>&1

# 等待执行完成
wait
check_status "fine"

# ===================== 汇总最终最优配置 =====================
echo -e "\n===== Summarize the final optimal configuration ====="
FINAL_BEST_FILE="${RESULT_DIR}/final_best_config.txt"
FINE_BEST_FILE="${RESULT_DIR}/best_config_fine.txt"

# 读取微调层最优RMSE
FINAL_BEST_RMSE=$(get_best_param "fine" "val_rmse")

# 生成最终最优配置文件
cat > ${FINAL_BEST_FILE} << EOF
图束模型全自动调优 - 最终最优配置
=====================================
调优完成时间：$(date +%Y-%m-%d\ %H:%M:%S)
最终最优验证集RMSE：${FINAL_BEST_RMSE}
=====================================
【核心参数】
lr: ${BEST_LR}
node_stalk_dim: ${BEST_NODE_DIM}
fusion_type: ${BEST_FUSION}
batch_size: ${BEST_BATCH_SIZE}

【次要参数】
edge_stalk_dim: ${BEST_EDGE_DIM}
num_diffusion_layers: ${BEST_LAYERS}
weight_decay: ${BEST_WD}
dropout_rate: ${BEST_DROPOUT}

【其他参数】
modals: ["text", "video"]
epochs: 30
=====================================
各阶段结果文件：
- 核心层：${RESULT_DIR}/best_config_core.txt
- 次要层：${RESULT_DIR}/best_config_secondary.txt
- 微调层：${RESULT_DIR}/best_config_fine.txt
- 超参影响分析图：${RESULT_DIR}/hyperparam_impact_*.png
EOF

echo "✅ The final optimal configuration has been saved.：${FINAL_BEST_FILE}"

# ===================== 调优完成 =====================
echo -e "\n===== Fully automatic optimization completed！====="
echo "end_time：$(date +%Y-%m-%d\ %H:%M:%S)"
echo "📊 All results files：${RESULT_DIR}"
echo "🏆 Ultimate optimal configuration：${FINAL_BEST_FILE}"
echo "📈 Hyperparameter Influence Analysis Diagram：${RESULT_DIR}/hyperparam_impact_core.png / secondary.png / fine.png"
echo "📄 logs：${LOG_DIR}"
