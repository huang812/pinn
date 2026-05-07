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
# 结果文件目录
# =========================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'SoH Estimation')
)

# =========================
# 模型与结果文件路径
# =========================
results_paths = {
    "GRU": os.path.join(BASE_DIR, "SoH_CaseA_Baseline.pth"),
    "LSTM": os.path.join(BASE_DIR, "SoH_CaseA_LSTM_Baseline.pth"),
    "Verhulst-Sum": os.path.join(BASE_DIR, "SoH_CaseA_Verhulst_Sum.pth"),
    "Verhulst-AdpBal": os.path.join(BASE_DIR, "SoH_CaseA_Verhulst_AdpBal.pth")
}

# =========================
# 检查文件是否存在
# =========================
for model_name, path in results_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Error] File does not exist: {path}\n"
            f"Please check BASE_DIR:\n{BASE_DIR}"
        )

# =========================
# 定义误差指标函数
# =========================
def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def calc_rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_pred - y_true) / (y_true + 1e-8)) ** 2))

# =========================
# 读取结果并计算误差
# =========================
models = []
rmse_values = []
mae_values = []
rmspe_values = []

for model_name, path in results_paths.items():
    results = torch.load(path, map_location="cpu", weights_only=False)

    y_true = np.array(results["U_true"]).squeeze()
    y_pred = np.array(results["U_pred"]).squeeze()

    # 若长度不一致，取最短长度
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = calc_rmse(y_true, y_pred)
    mae = calc_mae(y_true, y_pred)
    rmspe = calc_rmspe(y_true, y_pred)

    models.append(model_name)
    rmse_values.append(rmse)
    mae_values.append(mae)
    rmspe_values.append(rmspe)

# 转为 numpy 数组
rmse_values = np.array(rmse_values)
mae_values = np.array(mae_values)
rmspe_values = np.array(rmspe_values)

# =========================
# 控制台打印结果
# =========================
print("=" * 70)
print("Error Metrics of Different SOH Prediction Models")
print("=" * 70)
print(f"{'Model':20s} {'RMSE':>12s} {'MAE':>12s} {'RMSPE':>12s}")
print("-" * 70)
for i, model in enumerate(models):
    print(f"{model:20s} {rmse_values[i]:12.6f} {mae_values[i]:12.6f} {rmspe_values[i]:12.6f}")
print("=" * 70)

# =========================
# 创建三联柱状图
# =========================
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))

bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

metric_data = [rmse_values, mae_values, rmspe_values]
metric_names = ["RMSE", "MAE", "RMSPE"]

for ax, data, metric_name in zip(axes, metric_data, metric_names):
    bars = ax.bar(models, data, color=bar_colors, width=0.65, edgecolor="black", linewidth=0.8)

    ax.set_title(metric_name, fontsize=13)
    ax.set_ylabel("Error Value", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # x轴标签倾斜一点，防止重叠
    ax.tick_params(axis='x', rotation=15)

    # 在柱顶标数值
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

# =========================
# 保存图片
# =========================
save_path = os.path.join(BASE_DIR, "SoH_CaseA_Error_Bar_Comparison.png")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
print(f"\nFigure saved to: {save_path}")



plt.show()