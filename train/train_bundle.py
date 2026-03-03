import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
import argparse
from tqdm import tqdm
import torch.nn as nn

from data.preprocess_bundle import load_graph_bundle_data
from models.graph_bundle_rec import MultiModalGraphBundleRec


# Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Validation function (calculates the RMSE of the validation set/test set)
def evaluate_model(model, pairs, ratings, movie_feats, batch_size, device, criterion):
    model.eval()  # Evaluation mode (with Dropout/BatchNorm disabled)
    total_loss = 0.0
    num_batches = len(ratings) // batch_size
    with torch.no_grad():  # Disable gradients to speed up evaluation
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
    model.train()  # Return to training mode
    return avg_rmse


def train_single(config_path, test_mode=False):
    # 1. Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    set_seed(config["train"]["seed"])

    # Test mode parameters
    if test_mode:
        print("===== Bundle Model Testing Mode (Short Training)=====")
        config["train"]["epochs"] = config["train"]["test_epochs"]
        config["train"]["batch_size"] = config["train"]["test_batch_size"]
    else:
        print("===== Bundle Model Testing Mode =====")
    print(f"device：{device} | modals：{config['data']['modals']}")
    print(f"epochs：{config['train']['epochs']} | batch_size：{config['train']['batch_size']}")

    # 2. Load data (including training/validation/test sets)
    data_dict = load_graph_bundle_data(config["data"])

    # 3. Data moved to device
    movie_feats = {k: v.to(device) for k, v in data_dict["movie_feats"].items()}
    train_pairs = data_dict["train_pairs"].to(device)
    train_ratings = data_dict["train_ratings"].to(device)
    val_pairs = data_dict["val_pairs"].to(device)
    val_ratings = data_dict["val_ratings"].to(device)
    test_pairs = data_dict["test_pairs"].to(device)
    test_ratings = data_dict["test_ratings"].to(device)

    # 4. Force conversion of numerical parameters
    train_config = config["train"]
    model_config = config["model"]
    train_config["lr"] = float(train_config["lr"])
    train_config["batch_size"] = int(train_config["batch_size"])
    train_config["weight_decay"] = float(train_config["weight_decay"])
    train_config["epochs"] = int(train_config["epochs"])
    train_config["seed"] = int(train_config["seed"])
    model_config["node_stalk_dim"] = int(model_config["node_stalk_dim"])
    model_config["edge_stalk_dim"] = int(model_config["edge_stalk_dim"])
    model_config["num_diffusion_layers"] = int(model_config["num_diffusion_layers"])

    # 5. Initialize the model
    model_config_dict = {
        "num_users": data_dict["num_users"],
        "num_movies": data_dict["num_movies"],
        "modals": data_dict["modals"],
        "modal_in_dims": data_dict["modal_in_dims"],
        "node_stalk_dim": model_config["node_stalk_dim"],
        "edge_stalk_dim": model_config["edge_stalk_dim"],
        "num_diffusion_layers": model_config["num_diffusion_layers"],
        "fusion_type": model_config["fusion_type"]
    }
    model = MultiModalGraphBundleRec(model_config_dict, device=device).to(device)

    # 6. Optimizer and Loss Function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"]
    )
    criterion = nn.MSELoss()

    # 7. Early stop configuration (core: stop if the validation set RMSE continues to rise).
    best_val_rmse = float('inf')
    patience = 5  # If the RMSE does not decrease after 5 consecutive rounds of validation, stop early.
    patience_counter = 0
    best_model_state = None

    # 8. Training loop (with validation + early stop)
    model.train()
    for epoch in range(train_config["epochs"]):
        epoch_loss = 0.0
        # Randomly shuffle the training set
        perm = torch.randperm(len(train_ratings))
        num_batches = len(train_ratings) // train_config["batch_size"]

        # Training batch
        with tqdm(total=num_batches, desc=f"Epoch [{epoch + 1}/{train_config['epochs']}]") as pbar:
            for i in range(num_batches):
                start = i * train_config["batch_size"]
                end = start + train_config["batch_size"]
                batch_user = train_pairs[perm[start:end], 0]
                batch_movie = train_pairs[perm[start:end], 1]
                batch_rating = train_ratings[perm[start:end]]

                pred_rating = model(batch_user, batch_movie, movie_feats)
                loss = criterion(pred_rating.squeeze(), batch_rating)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Train Loss": loss.item(), "Train RMSE": torch.sqrt(loss).item()})

        # Calculate the RMSE of the training set in this round.
        train_rmse = torch.sqrt(torch.tensor(epoch_loss / num_batches)).item()

        # Calculate the RMSE of the validation set in this round (core: validation phase).
        val_rmse = evaluate_model(
            model, val_pairs, val_ratings, movie_feats,
            train_config["batch_size"], device, criterion
        )

        # Print the results of this round
        print(f"\nEpoch [{epoch + 1}] Summary:")
        print(f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        # Early Stop Logic
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # Save the optimal model (the model that performs best on the validation set).
            best_model_state = model.state_dict()
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": best_model_state,
                "config": model_config_dict,
                "best_val_rmse": best_val_rmse
            }, "checkpoints/best_graph_bundle_model.pth")
            print(f"✅ Improved RMSE on validation set! Saved optimal model.（Val RMSE: {best_val_rmse:.4f}）")
        else:
            patience_counter += 1
            print(f"⚠️  Validation set RMSE not improved (continuous{patience_counter}/{patience}）")
            if patience_counter >= patience:
                print(f"🛑 Early stop triggering! Optimal validation set RMSE：{best_val_rmse:.4f}")
                break

    # 9. Final evaluation (testing the test set with the optimal model)
    print("\n===== Final Test Set Evaluation =====")
    # Loading the optimal model
    model.load_state_dict(best_model_state)
    test_rmse = evaluate_model(
        model, test_pairs, test_ratings, movie_feats,
        train_config["batch_size"], device, criterion
    )
    print(f"Best Val RMSE: {best_val_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    # Save the final result
    with open("experiment_results/final_evaluation.txt", "w") as f:
        f.write(f"Best Validation RMSE: {best_val_rmse:.4f}\n")
        f.write(f"Test RMSE: {test_rmse:.4f}\n")
        f.write(f"Model Config: {model_config_dict}\n")

    print(f"\nTraining complete! Final results saved to：experiment_results/final_evaluation.txt")
    print(f"Optimal Model Path：checkpoints/best_graph_bundle_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of graph models (with validation and early stopping)")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--test-mode", action="store_true", help="Test mode (short training)")
    args = parser.parse_args()
    train_single(args.config, test_mode=args.test_mode)
