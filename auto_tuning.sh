#!/bin/bash
# Script for Fully Automated Hierarchical Hyperparameter Tuning of Bundle Models
# Usage Instructions: ./auto_tuning.sh [configuration file path]
# Example: ./auto_tuning.sh config/config.yaml

# ===================== Configuration items =====================
# Default configuration file path
CONFIG_PATH=${1:-"config/config.yaml"}
# Results Save Directory
RESULT_DIR="experiment_results"
# Python script path
TUNING_SCRIPT="train/hyperparam_tuning_bundle.py"
# Log directory
LOG_DIR="${RESULT_DIR}/auto_tuning_logs"

# ===================== utility functions =====================
# Read the optimal configuration function (extract specified parameters from a txt file).
get_best_param() {
    local stage=$1
    local param=$2
    local best_file="${RESULT_DIR}/best_config_${stage}.txt"
    # Extract parameter values (compatible with both numeric and string values)
    grep "^${param}:" ${best_file} | awk -F': ' '{print $2}' | tr -d '"'
}

# Check script execution status
check_status() {
    if [ $? -ne 0 ]; then
        echo "❌ error: $1 stage failed！"
        exit 1
    fi
}

# ===================== initialization =====================
echo "===== Fully Automated Layered Hyperparameter Tuning of the Bundle Model ====="
echo "start_time：$(date +%Y-%m-%d\ %H:%M:%S)"
echo "config：${CONFIG_PATH}"
echo "result_dir：${RESULT_DIR}"

# Create directory
mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

# Check necessary documents
if [ ! -f ${CONFIG_PATH} ]; then
    echo "❌ error：config ${CONFIG_PATH} not exist！"
    exit 1
fi

if [ ! -f ${TUNING_SCRIPT} ]; then
    echo "❌ error：sh ${TUNING_SCRIPT} not exist！"
    exit 1
fi

# ===================== Phase 1: Core Layer Optimization =====================
echo -e "\n===== Start core stage tuning.====="
echo "goal：lr、node_stalk_dim、fusion_type、batch_size"
echo "log：${LOG_DIR}/core_tuning.log"

# Core layer optimization (background operation + log saving)
nohup python ${TUNING_SCRIPT} \
    --config ${CONFIG_PATH} \
    --stage core > ${LOG_DIR}/core_tuning.log 2>&1

# Waiting for execution to complete
wait
check_status "core"

# Read the optimal configuration of the core layer
echo -e "\n===== Extract the optimal configuration of the core ====="
BEST_LR=$(get_best_param "core" "lr")
BEST_NODE_DIM=$(get_best_param "core" "node_stalk_dim")
BEST_FUSION=$(get_best_param "core" "fusion_type")
BEST_BATCH_SIZE=$(get_best_param "core" "batch_size")

# Verify extraction results
if [ -z "${BEST_LR}" ] || [ -z "${BEST_NODE_DIM}" ] || [ -z "${BEST_FUSION}" ] || [ -z "${BEST_BATCH_SIZE}" ]; then
    echo "❌ Error: Failed to extract optimal configuration for core stage！"
    exit 1
fi

echo "best config of core："
echo "  lr: ${BEST_LR}"
echo "  node_stalk_dim: ${BEST_NODE_DIM}"
echo "  fusion_type: ${BEST_FUSION}"
echo "  batch_size: ${BEST_BATCH_SIZE}"

# ===================== Phase 2: Secondary Layer Optimization =====================
echo -e "\n===== Start secondary stage tuning ====="
echo "goal：edge_stalk_dim、num_diffusion_layers、weight_decay、dropout_rate"
echo "based on ：lr=${BEST_LR}, node_dim=${BEST_NODE_DIM}, fusion=${BEST_FUSION}, bs=${BEST_BATCH_SIZE}"
echo "log：${LOG_DIR}/secondary_tuning.log"

# Run secondary layer tuning (passing in the optimal configuration of the core layer).
nohup python ${TUNING_SCRIPT} \
    --config ${CONFIG_PATH} \
    --stage secondary \
    --best_lr ${BEST_LR} \
    --best_node_dim ${BEST_NODE_DIM} \
    --best_fusion ${BEST_FUSION} \
    --best_batch_size ${BEST_BATCH_SIZE} > ${LOG_DIR}/secondary_tuning.log 2>&1

# Waiting for execution to complete
wait
check_status "Secondary"

# Read the optimal configuration of the secondary layer
echo -e "\n===== Extracting the optimal configuration of the secondary stage ====="
BEST_EDGE_DIM=$(get_best_param "secondary" "edge_stalk_dim")
BEST_LAYERS=$(get_best_param "secondary" "num_diffusion_layers")
BEST_WD=$(get_best_param "secondary" "weight_decay")
BEST_DROPOUT=$(get_best_param "secondary" "dropout_rate")

# Verify extraction results
if [ -z "${BEST_EDGE_DIM}" ] || [ -z "${BEST_LAYERS}" ] || [ -z "${BEST_WD}" ] || [ -z "${BEST_DROPOUT}" ]; then
    echo "❌ error：Secondary optimal configuration extraction failed.！"
    exit 1
fi

echo "Optimal configuration of secondary："
echo "  edge_stalk_dim: ${BEST_EDGE_DIM}"
echo "  num_diffusion_layers: ${BEST_LAYERS}"
echo "  weight_decay: ${BEST_WD}"
echo "  dropout_rate: ${BEST_DROPOUT}"

# ===================== Phase 3: Fine-tuning and optimization =====================
echo -e "\n===== Start fine-tuning optimization====="
echo "goal: Fine-tune the nearest values of core parameters"
echo "log：${LOG_DIR}/fine_tuning.log"

# Run fine-tuning layer optimization (pass in the optimal configuration for core and secondary layers)
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

# Waiting for execution to complete
wait
check_status "fine"

# ===================== Summarize the final optimal configuration =====================
echo -e "\n===== Summarize the final optimal configuration ====="
FINAL_BEST_FILE="${RESULT_DIR}/final_best_config.txt"
FINE_BEST_FILE="${RESULT_DIR}/best_config_fine.txt"

# Read the optimal RMSE of the fine-tuning layer
FINAL_BEST_RMSE=$(get_best_param "fine" "val_rmse")

# Generate the final optimal configuration file
cat > ${FINAL_BEST_FILE} << EOF
Fully Automated Tuning of Stalk Model - Final Optimal Configuration
===================================== Tuning Completion Time: $(date +%Y-%m-%d\ %H:%M:%S)
Final Optimal Validation Set RMSE: ${FINAL_BEST_RMSE}
=====================================
【Core Parameters】
lr: ${BEST_LR}
node_stalk_dim: ${BEST_NODE_DIM}
fusion_type: ${BEST_FUSION}
batch_size: ${BEST_BATCH_SIZE}
【Secondary Parameters】
edge_stalk_dim: ${BEST_EDGE_DIM}
num_diffusion_layers: ${BEST_LAYERS}
weight_decay: ${BEST_WD}
dropout_rate: ${BEST_DROPOUT}
【Other Parameters】
modals: ["text", "video"]
epochs: 30
======================================== Results files for each stage:
- Core layer: ${RESULT_DIR}/best_config_core.txt
- Secondary layer: ${RESULT_DIR}/best_config_secondary.txt
- Fine-tuning layer: ${RESULT_DIR}/best_config_fine.txt
- Hyperparameter impact analysis diagram: ${RESULT_DIR}/hyperparam_impact_*.png
EOF

echo "✅ The final optimal configuration has been saved.：${FINAL_BEST_FILE}"

# ===================== 调优完成 =====================
echo -e "\n===== Fully automatic optimization completed！====="
echo "end_time：$(date +%Y-%m-%d\ %H:%M:%S)"
echo "📊 All results files：${RESULT_DIR}"
echo "🏆 Ultimate optimal configuration：${FINAL_BEST_FILE}"
echo "📈 Hyperparameter Influence Analysis Diagram：${RESULT_DIR}/hyperparam_impact_core.png / secondary.png / fine.png"
echo "📄 logs：${LOG_DIR}"
