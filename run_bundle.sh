#!/bin/bash
# One-click script for hyperparameter tuning of bundle models
# Instructions:
# 1. Test mode (quick verification): ./run_bundle.sh test
# 2. Formal tuning (complete 100 configurations): ./run_bundle.sh
# 3. Train only the optimal configuration: ./run_bundle.sh train

# Configure environment variables (to avoid CUDA-related issues)
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/li/projects/MMSRec"

# Configuration file path
CONFIG_PATH="config/config.yaml"

# Log directory (automatically created)
LOG_DIR="experiment_results/logs"
mkdir -p ${LOG_DIR}

# Execute different logic based on parameters
if [ "$1" = "test" ]; then
    # Test mode: Short training + rapid validation
    echo "===== Start the beam model test mode ====="
    python test_short_train.py > ${LOG_DIR}/test_mode.log 2>&1
    echo "✅ Test mode complete！log：${LOG_DIR}/test_mode.log"

elif [ "$1" = "train" ]; then
    # Only train the optimal configuration (use after tuning).
    echo "===== Start optimal configuration and complete training ====="
    python train/train_bundle.py --config ${CONFIG_PATH} > ${LOG_DIR}/best_train.log 2>&1
    echo "✅ Optimal configuration training completed！日志：${LOG_DIR}/best_train.log"

else
    # Formal tuning mode: Complete hyperparameter search + visualization
    echo "===== Start hyperparameter tuning of the beam model (approximately 100 configurations).====="
    nohup python train/hyperparam_tuning_bundle.py --config ${CONFIG_PATH} > ${LOG_DIR}/tuning.log 2>&1 &
    echo "✅ Hyperparameter tuning has been started in the background.！"
    echo "📄 log files：${LOG_DIR}/tuning.log"
    echo "🔍 Check progress：tail -f ${LOG_DIR}/tuning.log"
fi
