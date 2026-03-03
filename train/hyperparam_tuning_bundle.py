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

# ===================== Global configuration (tuning strategy) =====================
# Optimization phase: "core" / "secondary" / "fine"
TUNING_STAGE = "core"
# Optimal configuration for the core layer
BEST_CORE_CONFIG = {
    "lr": 5e-4,
    "node_stalk_dim": 128,
    "fusion_type": "attention",
    "batch_size": 256
}


# ===================== utility functions =====================
def set_seed(seed=42):
    """Set a random seed to ensure the experiment is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, pairs, ratings, movie_feats, batch_size, device, criterion):
    """Evaluation function: Calculate the RMSE of the validation set/test set."""
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
    """The corresponding search space is returned based on the optimization phase."""
    if stage == "core":
        # lr、node_stalk_dim、fusion_type、batch_size
        return {
            "modals": [["text", "video"]],
            "lr": [1e-4, 3e-4, 5e-4, 8e-4, 1e-3],
            "node_stalk_dim": [32, 64, 128, 256],
            "fusion_type": ["attention", "concat", "average"],
            "batch_size": [64, 128, 256],
            # Fixed secondary parameters
            "edge_stalk_dim": [64],
            "num_diffusion_layers": [2],
            "weight_decay": [1e-4],
            "dropout_rate": [0.1],
            "epochs": [30]
        }
    elif stage == "secondary":
        # Secondary layer: Based on the optimal configuration of the core layer, optimize regularization, number of diffusion layers, and edge stem dimension.
        return {
            "modals": [["text", "video"]],
            "lr": [BEST_CORE_CONFIG["lr"]],
            "node_stalk_dim": [BEST_CORE_CONFIG["node_stalk_dim"]],
            "fusion_type": [BEST_CORE_CONFIG["fusion_type"]],
            "batch_size": [BEST_CORE_CONFIG["batch_size"]],
            # Secondary parameters to be optimized
            "edge_stalk_dim": [32, 64, 128],
            "num_diffusion_layers": [1, 2, 3, 4],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "dropout_rate": [0.0, 0.1, 0.2, 0.3],
            "epochs": [30]
        }
    elif stage == "fine":
        # Fine-tuning layer: fine-tuning of core parameter neighbor values
        return {
            "modals": [["text", "video"]],
            "lr": [BEST_CORE_CONFIG["lr"] * 0.8, BEST_CORE_CONFIG["lr"], BEST_CORE_CONFIG["lr"] * 1.2],
            "node_stalk_dim": [max(32, BEST_CORE_CONFIG["node_stalk_dim"] - 64),
                               BEST_CORE_CONFIG["node_stalk_dim"],
                               BEST_CORE_CONFIG["node_stalk_dim"] + 64],
            "fusion_type": [BEST_CORE_CONFIG["fusion_type"]],
            "batch_size": [BEST_CORE_CONFIG["batch_size"]],
            # Secondary parameters use the optimal value
            "edge_stalk_dim": [BEST_SECONDARY_CONFIG["edge_stalk_dim"]],
            "num_diffusion_layers": [BEST_SECONDARY_CONFIG["num_diffusion_layers"]],
            "weight_decay": [BEST_SECONDARY_CONFIG["weight_decay"]],
            "dropout_rate": [BEST_SECONDARY_CONFIG["dropout_rate"]],
            "epochs": [30]
        }
    else:
        raise ValueError(f"Ineffective tuning stage:{stage}，可选：core/secondary/fine")


def train_single_config(config, data_dict, device):
    """Single-set hyperparameter training (with early stop)"""
    # Type conversion
    for k in ["lr", "batch_size", "node_stalk_dim", "edge_stalk_dim",
              "num_diffusion_layers", "weight_decay", "epochs"]:
        config[k] = float(config[k]) if "lr" in k or "weight" in k else int(config[k])

    # Data moved to device
    movie_feats = {k: v.to(device) for k, v in data_dict["movie_feats"].items()}
    train_pairs = data_dict["train_pairs"].to(device)
    train_ratings = data_dict["train_ratings"].to(device)
    val_pairs = data_dict["val_pairs"].to(device)
    val_ratings = data_dict["val_ratings"].to(device)

    # Model Configuration
    model_config = {
        "num_users": data_dict["num_users"],
        "num_movies": data_dict["num_movies"],
        "modals": config["modals"],
        "modal_in_dims": data_dict["modal_in_dims"],
        "node_stalk_dim": config["node_stalk_dim"],
        "edge_stalk_dim": config["edge_stalk_dim"],
        "num_diffusion_layers": config["num_diffusion_layers"],
        "fusion_type": config["fusion_type"],
        "dropout_rate": config["dropout_rate"]
    }

    # Initialize the model
    model = MultiModalGraphBundleRec(model_config, device=device).to(device)

    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    criterion = nn.MSELoss()

    # Training + early stop
    best_val_rmse = float('inf')
    patience = 3  # During the optimization phase, patience is set to 5, balancing efficiency and effectiveness.
    patience_counter = 0

    model.train()
    for epoch in range(config["epochs"]):
        # Training batch
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

        # Validation set evaluation
        val_rmse = evaluate_model(
            model, val_pairs, val_ratings, movie_feats,
            config["batch_size"], device, criterion
        )

        # Early Stop Logic
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_rmse


def analyze_hyperparam_impact(results_df, stage):
    """Hyperparameter impact analysis visualization (core function)"""
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    if stage == "core":
        # Core layer: Analyzes lr, node_stalk_dim, fusion_type, and batch_size
        # 1. Learning rate vs RMSE
        lr_rmse = results_df.groupby("lr")["val_rmse"].agg(["mean", "std"])
        axes[0, 0].errorbar(lr_rmse.index, lr_rmse["mean"], yerr=lr_rmse["std"],
                            marker='o', linewidth=2, capsize=5)
        axes[0, 0].set_title("Learning rate vs RMSE", fontsize=12)
        axes[0, 0].set_xlabel("Learning rate")
        axes[0, 0].set_ylabel("Average Val RMSE")
        axes[0, 0].grid(alpha=0.3)

        # 2. Node stem space dimension vs RMSE
        dim_rmse = results_df.groupby("node_stalk_dim")["val_rmse"].agg(["mean", "std"])
        axes[0, 1].bar(dim_rmse.index.astype(str), dim_rmse["mean"],
                       yerr=dim_rmse["std"], capsize=5, color="#59a14f")
        axes[0, 1].set_title("Node stem space dimension vs RMSE", fontsize=12)
        axes[0, 1].set_xlabel("dimension")
        axes[0, 1].set_ylabel("Average Val RMSE")

        # 3. Fusion method vs RMSE
        fusion_rmse = results_df.groupby("fusion_type")["val_rmse"].agg(["mean", "std"])
        axes[1, 0].bar(fusion_rmse.index, fusion_rmse["mean"],
                       yerr=fusion_rmse["std"], capsize=5, color="#e15759")
        axes[1, 0].set_title("Fusion method vs RMSE", fontsize=12)
        axes[1, 0].set_xlabel("Fusion method")
        axes[1, 0].set_ylabel("Average Val RMSE")

        # 4. Batch size vs RMSE
        bs_rmse = results_df.groupby("batch_size")["val_rmse"].agg(["mean", "std"])
        axes[1, 1].bar(bs_rmse.index.astype(str), bs_rmse["mean"],
                       yerr=bs_rmse["std"], capsize=5, color="#f28e2b")
        axes[1, 1].set_title("Batch size vs RMSE", fontsize=12)
        axes[1, 1].set_xlabel("Batch size")
        axes[1, 1].set_ylabel("Average Val RMSE")

    elif stage == "secondary":
        # Secondary layer: Analyze the number of diffusion layers, stem dimension, regularization, and dropout.
        # 1. Number of diffusion layers vs. RMSE
        layer_rmse = results_df.groupby("num_diffusion_layers")["val_rmse"].agg(["mean", "std"])
        axes[0, 0].bar(layer_rmse.index.astype(str), layer_rmse["mean"],
                       yerr=layer_rmse["std"], capsize=5, color="#4e79a7")
        axes[0, 0].set_title("Number of diffusion layers vs. RMSE", fontsize=12)
        axes[0, 0].set_xlabel("layers")
        axes[0, 0].set_ylabel("Average Val RMSE")

        # 2. Piezometric Space Dimensions vs RMSE
        edge_dim_rmse = results_df.groupby("edge_stalk_dim")["val_rmse"].agg(["mean", "std"])
        axes[0, 1].bar(edge_dim_rmse.index.astype(str), edge_dim_rmse["mean"],
                       yerr=edge_dim_rmse["std"], capsize=5, color="#76b7b2")
        axes[0, 1].set_title("Piezometric Space Dimensions vs RMSE", fontsize=12)
        axes[0, 1].set_xlabel("Dimensions")
        axes[0, 1].set_ylabel("Average Val RMSE")

        # 3. L2 regularization vs RMSE
        wd_rmse = results_df.groupby("weight_decay")["val_rmse"].agg(["mean", "std"])
        axes[1, 0].errorbar(wd_rmse.index, wd_rmse["mean"], yerr=wd_rmse["std"],
                            marker='s', linewidth=2, capsize=5)
        axes[1, 0].set_title("L2 regularization vs RMSE", fontsize=12)
        axes[1, 0].set_xlabel("weight_decay")
        axes[1, 0].set_ylabel("Average Val RMSE")
        axes[1, 0].grid(alpha=0.3)

        # 4. Dropout rate vs RMSE
        dropout_rmse = results_df.groupby("dropout_rate")["val_rmse"].agg(["mean", "std"])
        axes[1, 1].errorbar(dropout_rmse.index, dropout_rmse["mean"], yerr=dropout_rmse["std"],
                            marker='^', linewidth=2, capsize=5)
        axes[1, 1].set_title("Dropout rate vs RMSE", fontsize=12)
        axes[1, 1].set_xlabel("dropout_rate")
        axes[1, 1].set_ylabel("Average Val RMSE")
        axes[1, 1].grid(alpha=0.3)

    # Save visualization
    plt.tight_layout()
    save_path = f"experiment_results/hyperparam_impact_{stage}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ The hyperparameter influence analysis diagram has been saved:{save_path}")


# ===================== Tuning the main function =====================
def hierarchical_tuning(config_path):
    """Layered hyperparameter tuning main function"""
    # 1. initialization
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    device = torch.device(base_config["train"]["device"] if torch.cuda.is_available() else "cpu")
    set_seed(base_config["train"]["seed"])
    os.makedirs("experiment_results", exist_ok=True)

    # 2. Loading data
    print(f"===== Loading Data ({TUNING_STAGE} Layer Tuning) =====")
    data_dict = load_graph_bundle_data(base_config["data"])

    # 3. Obtain the search space at the current stage
    search_space = get_search_space(TUNING_STAGE)
    print(f"Current optimization stage:{TUNING_STAGE}")
    print(f"Search space:{search_space.keys()}")

    # 4. Generate configurations (80 sets for the core layer, 48 sets for the secondary layer, and 9 sets for the fine-tuning layer).
    n_trials = {"core": 80, "secondary": 48, "fine": 9}[TUNING_STAGE]
    configs = []
    for _ in range(n_trials):
        config = {}
        for k, v_list in search_space.items():
            config[k] = random.choice(v_list)
        configs.append(config)
    print(f"Number of configurations generated:{len(configs)}")

    # 5. Execution tuning
    results = []
    best_rmse = float('inf')
    best_config = None

    for idx, config in enumerate(tqdm(configs, desc=f"{TUNING_STAGE} stage tuning")):
        try:
            start_time = datetime.now()
            val_rmse = train_single_config(config, data_dict, device)
            train_duration = (datetime.now() - start_time).total_seconds() / 60

            # Collection Results
            result_row = {
                "config_id": idx,
                "val_rmse": val_rmse,
                "train_duration_min": round(train_duration, 2),
                "is_best": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": ""
            }
            # Merge configuration parameters
            result_row.update(config)

            # Update to optimal configuration
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_config = config.copy()
                result_row["is_best"] = True

            results.append(result_row)
            tqdm.write(f"config{idx} | Val RMSE: {val_rmse:.4f} | best: {best_rmse:.4f} | time: {train_duration:.2f}min")

        except Exception as e:
            # Log failure configuration
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

    # 6. Save results
    results_df = pd.DataFrame(results)
    # Complete results
    results_df.to_csv(f"experiment_results/tuning_{TUNING_STAGE}_full.csv", index=False, encoding="utf-8")
    # Successful results
    success_df = results_df[results_df["val_rmse"].notna()]
    success_df.to_csv(f"experiment_results/tuning_{TUNING_STAGE}_success.csv", index=False, encoding="utf-8")
    # Optimal configuration
    with open(f"experiment_results/best_config_{TUNING_STAGE}.txt", "w") as f:
        f.write(f"Optimal configuration of the {TUNING_STAGE} stage\n")
        f.write(f"====================\n")
        f.write(f"best Val RMSE：{best_rmse:.4f}\n")
        f.write(f"Optimization completion time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"====================\n")
        if best_config:
            for k, v in best_config.items():
                f.write(f"{k}: {v}\n")

    # 7. Hyperparameter impact analysis visualization
    if not success_df.empty:
        analyze_hyperparam_impact(success_df, TUNING_STAGE)

    # 8. Output Summary
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
    # Optimal configuration parameters for the core layer
    parser.add_argument("--best_lr", type=float, default=5e-4, help="Core stage optimal lr")
    parser.add_argument("--best_node_dim", type=int, default=128, help="Core stage node_stalk_dim")
    parser.add_argument("--best_fusion", type=str, default="attention", help="Core stage fusion_type")
    parser.add_argument("--best_batch_size", type=int, default=256, help="Core stage batch_size")
    # Optimal configuration parameters for secondary layer
    parser.add_argument("--best_edge_dim", type=int, default=64, help="Secondary stage edge_stalk_dim")
    parser.add_argument("--best_layers", type=int, default=2, help="Secondary stage num_diffusion_layers")
    parser.add_argument("--best_wd", type=float, default=1e-4, help="Secondary stage weight_decay")
    parser.add_argument("--best_dropout", type=float, default=0.1, help="Secondary stage dropout_rate")

    args = parser.parse_args()

    # Update global configuration
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
