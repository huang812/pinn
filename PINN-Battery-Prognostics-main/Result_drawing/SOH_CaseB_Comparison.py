import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch

# =========================
# Matplotlib 全局设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 颜色与线型设置（论文简洁风格）
# =========================
COLORS = {
    "Ground Truth": "black",
    "GRU": "#1f77b4",
    "LSTM": "#2ca02c",
    "Verhulst-Sum": "#ff7f0e",
    "Verhulst-AdpBal": "#d62728"
}

LINESTYLES = {
    "Ground Truth": "-",
    "GRU": "--",
    "LSTM": "-.",
    "Verhulst-Sum": "--",
    "Verhulst-AdpBal": "-"
}

LINEWIDTHS = {
    "Ground Truth": 2.2,
    "GRU": 1.8,
    "LSTM": 1.8,
    "Verhulst-Sum": 1.8,
    "Verhulst-AdpBal": 2.0
}

# =========================
# 结果文件目录
# =========================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'SoH Estimation')
)

# =========================
# Case B 结果文件路径
# =========================
results_paths = {
    "GRU": os.path.join(BASE_DIR, "SoH_CaseB_Baseline.pth"),
    "LSTM": os.path.join(BASE_DIR, "SoH_CaseB_LSTM_Baseline.pth"),
    "Verhulst-Sum": os.path.join(BASE_DIR, "SoH_CaseB_Verhulst_Sum.pth"),
    "Verhulst-AdpBal": os.path.join(BASE_DIR, "SoH_CaseB_Verhulst_AdpBal.pth")
}

# Ground Truth 从任意一个结果文件中读取 U_true，这里用 GRU 文件
ground_truth_path = results_paths["GRU"]

# =========================
# 检查文件是否存在
# =========================
all_paths = {"Ground Truth": ground_truth_path}
all_paths.update(results_paths)

for model_name, path in all_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Error] File does not exist: {path}\n"
            f"Please check BASE_DIR:\n{BASE_DIR}"
        )

# =========================
# 读取真实值
# =========================
ground_truth_results = torch.load(
    ground_truth_path,
    map_location="cpu",
    weights_only=False
)

cycles = np.array(ground_truth_results["Cycles"]).squeeze()
ground_truth = np.array(ground_truth_results["U_true"]).squeeze()

# 排序，防止横坐标乱序
sort_idx = np.argsort(cycles)
cycles = cycles[sort_idx]
ground_truth = ground_truth[sort_idx]

# =========================
# 创建图形
# =========================
plt.figure(figsize=(8.5, 5.8))

# 绘制真实值
plt.plot(
    cycles,
    ground_truth,
    label="Ground Truth",
    color=COLORS["Ground Truth"],
    linestyle=LINESTYLES["Ground Truth"],
    linewidth=LINEWIDTHS["Ground Truth"]
)

# =========================
# RMSE 计算与曲线绘制
# =========================
rmse_values = {}

for model_name, path in results_paths.items():
    results = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    u_pred = np.array(results["U_pred"]).squeeze()
    model_cycles = np.array(results["Cycles"]).squeeze()

    # 排序
    model_sort_idx = np.argsort(model_cycles)
    model_cycles = model_cycles[model_sort_idx]
    u_pred = u_pred[model_sort_idx]

    # 长度保护
    min_len = min(len(cycles), len(model_cycles), len(ground_truth), len(u_pred))
    cycles_plot = cycles[:min_len]
    ground_truth_plot = ground_truth[:min_len]
    u_pred_plot = u_pred[:min_len]

    rmse = np.sqrt(np.mean((u_pred_plot - ground_truth_plot) ** 2))
    rmse_values[model_name] = rmse

    plt.plot(
        cycles_plot,
        u_pred_plot,
        label=model_name,
        color=COLORS[model_name],
        linestyle=LINESTYLES[model_name],
        linewidth=LINEWIDTHS[model_name]
    )

# =========================
# 图表设置
# =========================
plt.xlabel("Cycles", fontsize=13)
plt.ylabel("SOH", fontsize=13)
plt.xlim(left=0)
plt.ylim(0.75, 1.02)   # 可根据 Case B 的结果再微调
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=10, loc="upper right", frameon=True)
plt.tight_layout()

# =========================
# 保存图片
# =========================
save_path = os.path.join(BASE_DIR, "SoH_CaseB_Prediction_Comparison_Simplified.png")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

# =========================
# 打印 RMSE
# =========================
print("\nRMSE Results (Case B):")
for model, rmse in rmse_values.items():
    print(f"{model}: {rmse:.4f}")

best_model = min(rmse_values, key=rmse_values.get)
print(f"\nBest Model: {best_model}")
print(f"Lowest RMSE: {rmse_values[best_model]:.4f}")

plt.show()