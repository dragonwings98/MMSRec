import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 1. Read tuning data =====================
# Core layer optimization data (including configurations for all fusion methods)
core_data = pd.read_csv("experiment_results/tuning_secondary_success.csv")

# Filter data related to fusion methods (keep only key columns).
fusion_data = core_data[["dropout_rate", "val_rmse"]]

# ===================== 2. Statistical key data =====================
# 2.1 The optimal RMSE (minimum value) for each fusion method.
fusion_best = fusion_data.groupby("dropout_rate")["val_rmse"].min()
# 2.2 Average RMSE (stability) for each fusion method
fusion_avg = fusion_data.groupby("dropout_rate")["val_rmse"].mean()

# Printout results (corresponding to Experiment Report 7.2)
print("===== 7.2 Comparison of fusion methods =====")
print("【Optimal RMSE (Core Comparison)】")
print(fusion_best)
print("\n【Average RMSE (stability comparison)】")
print(fusion_avg)

# ===================== 3. Visualization=====================
# Set chart style
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))

# Plot a bar chart comparing fusion methods versus optimal RMSE.
ax = sns.barplot(x=fusion_best.index, y=fusion_best.values, palette="Set2")

# Add numerical labels
for i, v in enumerate(fusion_best.values):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=12)

# Set titles and tags
plt.title("dropout_rate vs Validation RMSE", fontsize=14)
plt.xlabel("dropout_rate", fontsize=12)
plt.ylabel("Validation RMSE (Lower is Better)", fontsize=12)
plt.ylim(0.85, 0.92)  # Define the y-axis range to highlight differences

# Save the charts
plt.savefig("experiment_results/dropout_rate_rmse.png", dpi=300, bbox_inches="tight")
print("\n✅ Save the charts ：experiment_results/dropout_rate_rmse.png")
