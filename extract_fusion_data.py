import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 1. 读取调优数据 =====================
# 核心层调优数据（包含所有融合方式的配置）
core_data = pd.read_csv("experiment_results/tuning_secondary_success.csv")

# 筛选融合方式相关数据（只保留关键列）
fusion_data = core_data[["dropout_rate", "val_rmse"]]

# ===================== 2. 统计关键数据 =====================
# 2.1 每种融合方式的最优RMSE（最小值）
fusion_best = fusion_data.groupby("dropout_rate")["val_rmse"].min()
# 2.2 每种融合方式的平均RMSE（稳定性）
fusion_avg = fusion_data.groupby("dropout_rate")["val_rmse"].mean()

# 打印结果（对应实验报告7.2）
print("===== 7.2 融合方式对比数据 =====")
print("【最优RMSE（核心对比）】")
print(fusion_best)
print("\n【平均RMSE（稳定性对比）】")
print(fusion_avg)

# ===================== 3. 可视化（生成论文级图表） =====================
# 设置图表样式
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))

# 绘制融合方式vs最优RMSE的柱状图
ax = sns.barplot(x=fusion_best.index, y=fusion_best.values, palette="Set2")

# 添加数值标签
for i, v in enumerate(fusion_best.values):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=12)

# 设置标题和标签
plt.title("dropout_rate vs Validation RMSE", fontsize=14)
plt.xlabel("dropout_rate", fontsize=12)
plt.ylabel("Validation RMSE (Lower is Better)", fontsize=12)
plt.ylim(0.85, 0.92)  # 限定y轴范围，突出差异

# 保存图表（可直接用于论文/报告）
plt.savefig("experiment_results/dropout_rate_rmse.png", dpi=300, bbox_inches="tight")
print("\n✅ 融合方式对比图表已保存：experiment_results/dropout_rate_rmse.png")
