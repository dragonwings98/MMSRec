import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
from data.preprocess_bundle import load_graph_bundle_data
from models.graph_bundle_rec import MultiModalGraphBundleRec

# ===================== 全局配置（调优策略） =====================
# 调优阶段："core"（核心层）/"secondary"（次要层）/"fine"（微调层）
TUNING_STAGE = "core"
# 核心层最优配置（跑完core后手动填入，供secondary/fine层使用）
BEST_CORE_CONFIG = {
    "lr": 5e-4,
    "node_stalk_dim": 128,
    "fusion_type": "attention",
    "batch_size": 256
}


# ===================== 工具函数 =====================
def set_seed(seed=42):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, pairs, ratings, movie_feats, batch_size, device, criterion):
    """评估函数：计算验证集/测试集RMSE"""
    model.eval()
    total_loss = 0.0
    num_batches = len(ratings) // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_user = pairs[start:end, 0].to(device)
            batch_movie = pairs[start:end, 1].to(device)
            batch_rating = ratings[start:end].to(device)

            pred_rating = model(batch_user, batch_movie, movie_feats)
            loss = criterion(pred_rating.squeeze(), batch_rating)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    avg_rmse = torch.sqrt(torch.tensor(avg_loss)).item()
    model.train()
    return avg_rmse


def get_search_space(stage):
    """根据调优阶段返回对应的搜索空间"""
    if stage == "core":
        # 核心层：快速筛选lr、node_stalk_dim、fusion_type、batch_size
        return {
            "modals": [["text", "video"]],  # 先固定多模态（重点优化）
            "lr": [1e-4, 3e-4, 5e-4, 8e-4, 1e-3],
            "node_stalk_dim": [32, 64, 128, 256],
            "fusion_type": ["attention", "concat", "average"],
            "batch_size": [64, 128, 256],
            # 固定次要参数
            "edge_stalk_dim": [64],
            "num_diffusion_layers": [2],
            "weight_decay": [1e-4],
            "dropout_rate": [0.1],
            "epochs": [30]
        }
    elif stage == "secondary":
        # 次要层：基于核心层最优配置，优化正则化、扩散层数、边茎维度
        return {
            "modals": [["text", "video"]],
            "lr": [BEST_CORE_CONFIG["lr"]],
            "node_stalk_dim": [BEST_CORE_CONFIG["node_stalk_dim"]],
            "fusion_type": [BEST_CORE_CONFIG["fusion_type"]],
            "batch_size": [BEST_CORE_CONFIG["batch_size"]],
            # 待优化的次要参数
            "edge_stalk_dim": [32, 64, 128],
            "num_diffusion_layers": [1, 2, 3, 4],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "dropout_rate": [0.0, 0.1, 0.2, 0.3],
            "epochs": [30]
        }
    elif stage == "fine":
        # 微调层：核心参数邻近值微调
        return {
            "modals": [["text", "video"]],
            "lr": [BEST_CORE_CONFIG["lr"] * 0.8, BEST_CORE_CONFIG["lr"], BEST_CORE_CONFIG["lr"] * 1.2],
            "node_stalk_dim": [max(32, BEST_CORE_CONFIG["node_stalk_dim"] - 64),
                               BEST_CORE_CONFIG["node_stalk_dim"],
                               BEST_CORE_CONFIG["node_stalk_dim"] + 64],
            "fusion_type": [BEST_CORE_CONFIG["fusion_type"]],
            "batch_size": [BEST_CORE_CONFIG["batch_size"]],
            # 次要参数用最优值
            "edge_stalk_dim": [BEST_SECONDARY_CONFIG["edge_stalk_dim"]],
            "num_diffusion_layers": [BEST_SECONDARY_CONFIG["num_diffusion_layers"]],
            "weight_decay": [BEST_SECONDARY_CONFIG["weight_decay"]],
            "dropout_rate": [BEST_SECONDARY_CONFIG["dropout_rate"]],
            "epochs": [30]
        }
    else:
        raise ValueError(f"Ineffective tuning stage:{stage}，可选：core/secondary/fine")


def train_single_config(config, data_dict, device):
    """单组超参训练（带早停）"""
    # 类型转换
    for k in ["lr", "batch_size", "node_stalk_dim", "edge_stalk_dim",
              "num_diffusion_layers", "weight_decay", "epochs"]:
        config[k] = float(config[k]) if "lr" in k or "weight" in k else int(config[k])

    # 数据移到设备
    movie_feats = {k: v.to(device) for k, v in data_dict["movie_feats"].items()}
    train_pairs = data_dict["train_pairs"].to(device)
    train_ratings = data_dict["train_ratings"].to(device)
    val_pairs = data_dict["val_pairs"].to(device)
    val_ratings = data_dict["val_ratings"].to(device)

    # 模型配置
    model_config = {
        "num_users": data_dict["num_users"],
        "num_movies": data_dict["num_movies"],
        "modals": config["modals"],
        "modal_in_dims": data_dict["modal_in_dims"],
        "node_stalk_dim": config["node_stalk_dim"],
        "edge_stalk_dim": config["edge_stalk_dim"],
        "num_diffusion_layers": config["num_diffusion_layers"],
        "fusion_type": config["fusion_type"],
        "dropout_rate": config["dropout_rate"]  # 新增dropout配置
    }

    # 初始化模型
    model = MultiModalGraphBundleRec(model_config, device=device).to(device)

    # 优化器与损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    criterion = nn.MSELoss()

    # 训练+早停
    best_val_rmse = float('inf')
    patience = 3  # 调优阶段patience=5，平衡效率和效果
    patience_counter = 0

    model.train()
    for epoch in range(config["epochs"]):
        # 训练批次
        epoch_loss = 0.0
        perm = torch.randperm(len(train_ratings))
        num_batches = len(train_ratings) // config["batch_size"]

        for i in range(num_batches):
            start = i * config["batch_size"]
            end = start + config["batch_size"]
            batch_user = train_pairs[perm[start:end], 0]
            batch_movie = train_pairs[perm[start:end], 1]
            batch_rating = train_ratings[perm[start:end]]

            pred_rating = model(batch_user, batch_movie, movie_feats)
            loss = criterion(pred_rating.squeeze(), batch_rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证集评估
        val_rmse = evaluate_model(
            model, val_pairs, val_ratings, movie_feats,
            config["batch_size"], device, criterion
        )

        # 早停逻辑
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_rmse


def analyze_hyperparam_impact(results_df, stage):
    """超参影响分析可视化（核心功能）"""
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    if stage == "core":
        # 核心层：分析lr、node_stalk_dim、fusion_type、batch_size
        # 1. 学习率 vs RMSE
        lr_rmse = results_df.groupby("lr")["val_rmse"].agg(["mean", "std"])
        axes[0, 0].errorbar(lr_rmse.index, lr_rmse["mean"], yerr=lr_rmse["std"],
                            marker='o', linewidth=2, capsize=5)
        axes[0, 0].set_title("Learning rate vs RMSE", fontsize=12)
        axes[0, 0].set_xlabel("Learning rate")
        axes[0, 0].set_ylabel("Average Val RMSE")
        axes[0, 0].grid(alpha=0.3)

        # 2. 节点茎空间维度 vs RMSE
        dim_rmse = results_df.groupby("node_stalk_dim")["val_rmse"].agg(["mean", "std"])
        axes[0, 1].bar(dim_rmse.index.astype(str), dim_rmse["mean"],
                       yerr=dim_rmse["std"], capsize=5, color="#59a14f")
        axes[0, 1].set_title("Node stem space dimension vs RMSE", fontsize=12)
        axes[0, 1].set_xlabel("dimension")
        axes[0, 1].set_ylabel("Average Val RMSE")

        # 3. 融合方式 vs RMSE
        fusion_rmse = results_df.groupby("fusion_type")["val_rmse"].agg(["mean", "std"])
        axes[1, 0].bar(fusion_rmse.index, fusion_rmse["mean"],
                       yerr=fusion_rmse["std"], capsize=5, color="#e15759")
        axes[1, 0].set_title("Fusion method vs RMSE", fontsize=12)
        axes[1, 0].set_xlabel("Fusion method")
        axes[1, 0].set_ylabel("Average Val RMSE")

        # 4. 批次大小 vs RMSE
        bs_rmse = results_df.groupby("batch_size")["val_rmse"].agg(["mean", "std"])
        axes[1, 1].bar(bs_rmse.index.astype(str), bs_rmse["mean"],
                       yerr=bs_rmse["std"], capsize=5, color="#f28e2b")
        axes[1, 1].set_title("Batch size vs RMSE", fontsize=12)
        axes[1, 1].set_xlabel("Batch size")
        axes[1, 1].set_ylabel("Average Val RMSE")

    elif stage == "secondary":
        # 次要层：分析扩散层数、边茎维度、正则化、dropout
        # 1. 图扩散层数 vs RMSE
        layer_rmse = results_df.groupby("num_diffusion_layers")["val_rmse"].agg(["mean", "std"])
        axes[0, 0].bar(layer_rmse.index.astype(str), layer_rmse["mean"],
                       yerr=layer_rmse["std"], capsize=5, color="#4e79a7")
        axes[0, 0].set_title("Number of diffusion layers vs. RMSE", fontsize=12)
        axes[0, 0].set_xlabel("layers")
        axes[0, 0].set_ylabel("Average Val RMSE")

        # 2. 边茎空间维度 vs RMSE
        edge_dim_rmse = results_df.groupby("edge_stalk_dim")["val_rmse"].agg(["mean", "std"])
        axes[0, 1].bar(edge_dim_rmse.index.astype(str), edge_dim_rmse["mean"],
                       yerr=edge_dim_rmse["std"], capsize=5, color="#76b7b2")
        axes[0, 1].set_title("Piezometric Space Dimensions vs RMSE", fontsize=12)
        axes[0, 1].set_xlabel("Dimensions")
        axes[0, 1].set_ylabel("Average Val RMSE")

        # 3. L2正则化 vs RMSE
        wd_rmse = results_df.groupby("weight_decay")["val_rmse"].agg(["mean", "std"])
        axes[1, 0].errorbar(wd_rmse.index, wd_rmse["mean"], yerr=wd_rmse["std"],
                            marker='s', linewidth=2, capsize=5)
        axes[1, 0].set_title("L2 regularization vs RMSE", fontsize=12)
        axes[1, 0].set_xlabel("weight_decay")
        axes[1, 0].set_ylabel("Average Val RMSE")
        axes[1, 0].grid(alpha=0.3)

        # 4. Dropout率 vs RMSE
        dropout_rmse = results_df.groupby("dropout_rate")["val_rmse"].agg(["mean", "std"])
        axes[1, 1].errorbar(dropout_rmse.index, dropout_rmse["mean"], yerr=dropout_rmse["std"],
                            marker='^', linewidth=2, capsize=5)
        axes[1, 1].set_title("Dropout rate vs RMSE", fontsize=12)
        axes[1, 1].set_xlabel("dropout_rate")
        axes[1, 1].set_ylabel("Average Val RMSE")
        axes[1, 1].grid(alpha=0.3)

    # 保存可视化
    plt.tight_layout()
    save_path = f"experiment_results/hyperparam_impact_{stage}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ The hyperparameter influence analysis diagram has been saved:{save_path}")


# ===================== 调优主函数 =====================
def hierarchical_tuning(config_path):
    """分层超参调优主函数"""
    # 1. 初始化
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    device = torch.device(base_config["train"]["device"] if torch.cuda.is_available() else "cpu")
    set_seed(base_config["train"]["seed"])
    os.makedirs("experiment_results", exist_ok=True)

    # 2. 加载数据
    print(f"===== Loading Data ({TUNING_STAGE} Layer Tuning) =====")
    data_dict = load_graph_bundle_data(base_config["data"])

    # 3. 获取当前阶段的搜索空间
    search_space = get_search_space(TUNING_STAGE)
    print(f"Current optimization stage:{TUNING_STAGE}")
    print(f"Search space:{search_space.keys()}")

    # 4. 生成配置（核心层80组，次要层48组，微调层9组）
    n_trials = {"core": 80, "secondary": 48, "fine": 9}[TUNING_STAGE]
    configs = []
    for _ in range(n_trials):
        config = {}
        for k, v_list in search_space.items():
            config[k] = random.choice(v_list)
        configs.append(config)
    print(f"Number of configurations generated:{len(configs)}")

    # 5. 执行调优
    results = []
    best_rmse = float('inf')
    best_config = None

    for idx, config in enumerate(tqdm(configs, desc=f"{TUNING_STAGE} stage tuning")):
        try:
            start_time = datetime.now()
            val_rmse = train_single_config(config, data_dict, device)
            train_duration = (datetime.now() - start_time).total_seconds() / 60

            # 收集结果
            result_row = {
                "config_id": idx,
                "val_rmse": val_rmse,
                "train_duration_min": round(train_duration, 2),
                "is_best": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": ""
            }
            # 合并配置参数
            result_row.update(config)

            # 更新最优配置
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_config = config.copy()
                result_row["is_best"] = True

            results.append(result_row)
            tqdm.write(f"config{idx} | Val RMSE: {val_rmse:.4f} | best: {best_rmse:.4f} | time: {train_duration:.2f}min")

        except Exception as e:
            # 记录失败配置
            error_msg = str(e)[:100]
            results.append({
                "config_id": idx,
                "val_rmse": float('nan'),
                "train_duration_min": float('nan'),
                "is_best": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": error_msg,
                **config
            })
            print(f"❌ config{idx}failed：{e}")
            continue

    # 6. 保存结果
    results_df = pd.DataFrame(results)
    # 完整结果
    results_df.to_csv(f"experiment_results/tuning_{TUNING_STAGE}_full.csv", index=False, encoding="utf-8")
    # 成功结果
    success_df = results_df[results_df["val_rmse"].notna()]
    success_df.to_csv(f"experiment_results/tuning_{TUNING_STAGE}_success.csv", index=False, encoding="utf-8")
    # 最优配置
    with open(f"experiment_results/best_config_{TUNING_STAGE}.txt", "w") as f:
        f.write(f"Optimal configuration of the {TUNING_STAGE} stage\n")
        f.write(f"====================\n")
        f.write(f"best Val RMSE：{best_rmse:.4f}\n")
        f.write(f"Optimization completion time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"====================\n")
        if best_config:
            for k, v in best_config.items():
                f.write(f"{k}: {v}\n")

    # 7. 超参影响分析可视化
    if not success_df.empty:
        analyze_hyperparam_impact(success_df, TUNING_STAGE)

    # 8. 输出总结
    print(f"\n===== {TUNING_STAGE} optimization completed =====")
    print(f"📊 full result：experiment_results/tuning_{TUNING_STAGE}_full.csv")
    print(f"✅ success result：experiment_results/tuning_{TUNING_STAGE}_success.csv")
    print(f"🏆 best config：experiment_results/best_config_{TUNING_STAGE}.txt")
    print(f"🎯 best Val RMSE：{best_rmse:.4f}")
    print(f"====================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layered hyperparameter tuning of the bundle model")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--stage", default="core", choices=["core", "secondary", "fine"], help="Optimization stage")
    # 核心层最优配置参数
    parser.add_argument("--best_lr", type=float, default=5e-4, help="Core stage optimal lr")
    parser.add_argument("--best_node_dim", type=int, default=128, help="Core stage node_stalk_dim")
    parser.add_argument("--best_fusion", type=str, default="attention", help="Core stage fusion_type")
    parser.add_argument("--best_batch_size", type=int, default=256, help="Core stage batch_size")
    # 次要层最优配置参数
    parser.add_argument("--best_edge_dim", type=int, default=64, help="Secondary stage edge_stalk_dim")
    parser.add_argument("--best_layers", type=int, default=2, help="Secondary stage num_diffusion_layers")
    parser.add_argument("--best_wd", type=float, default=1e-4, help="Secondary stage weight_decay")
    parser.add_argument("--best_dropout", type=float, default=0.1, help="Secondary stage dropout_rate")

    args = parser.parse_args()

    # 更新全局配置
    TUNING_STAGE = args.stage
    BEST_CORE_CONFIG = {
        "lr": args.best_lr,
        "node_stalk_dim": args.best_node_dim,
        "fusion_type": args.best_fusion,
        "batch_size": args.best_batch_size
    }
    BEST_SECONDARY_CONFIG = {
        "edge_stalk_dim": args.best_edge_dim,
        "num_diffusion_layers": args.best_layers,
        "weight_decay": args.best_wd,
        "dropout_rate": args.best_dropout
    }

    hierarchical_tuning(args.config)
