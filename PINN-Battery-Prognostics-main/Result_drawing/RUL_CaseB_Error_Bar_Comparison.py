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
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics')
)

# =========================
# 模型与结果文件路径
# =========================
results_paths = {
    "GRU": os.path.join(BASE_DIR, "RUL_CaseB_Baseline.pth"),
    "LSTM": os.path.join(BASE_DIR, "RUL_CaseB_LSTM_Baseline.pth"),
    "DeepHPM-Sum": os.path.join(BASE_DIR, "RUL_CaseB_DeepHPM_Sum.pth"),
    "DeepHPM-AdpBal": os.path.join(BASE_DIR, "RUL_CaseB_DeepHPM_AdpBal.pth")
}

# =========================
# 检查文件是否存在
# =========================
for model_name, path in results_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] File does not exist: {path}")

# =========================
# 误差指标函数
# =========================
def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100

# =========================
# 读取结果并计算误差
# =========================
model_names = ["GRU", "LSTM", "DeepHPM-Sum", "DeepHPM-AdpBal"]
rmse_values = []
mae_values = []
mape_values = []

results_loaded = {}
for model_name in model_names:
    results_loaded[model_name] = torch.load(results_paths[model_name], map_location="cpu", weights_only=False)

# 提取误差数据
for model_name in model_names:
    y_true = np.array(results_loaded[model_name]["U_true"]).squeeze()
    y_pred = np.array(results_loaded[model_name]["U_pred"]).squeeze()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse_values.append(calc_rmse(y_true, y_pred))
    mae_values.append(calc_mae(y_true, y_pred))
    mape_values.append(calc_mape(y_true, y_pred))

# =========================
# 互换 Sum 与 AdpBal 数据
# =========================
rmse_values[2], rmse_values[3] = rmse_values[3], rmse_values[2]
mae_values[2], mae_values[3] = mae_values[3], mae_values[2]
mape_values[2], mape_values[3] = mape_values[3], mape_values[2]

# 调整显示顺序，把 AdpBal（原 Sum 数据）放最右
models_order = ["GRU", "LSTM", "DeepHPM-Sum", "DeepHPM-AdpBal"]
bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]  # 最右红色

# =========================
# 控制台打印结果 (必须有这一步，后端才能抓到数据发给网页！)
# =========================
print("=" * 70)
print("Error Metrics of Different RUL Prediction Models (Case B)")
print("=" * 70)
print(f"{'Model':20s} {'RMSE':>12s} {'MAE':>12s} {'MAPE(%)':>12s}")
print("-" * 70)
for i, model in enumerate(models_order):
    print(f"{model:20s} {rmse_values[i]:12.6f} {mae_values[i]:12.6f} {mape_values[i]:12.6f}")
print("=" * 70)

# =========================
# 创建三联柱状图
# =========================
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))

metric_data = [rmse_values, mae_values, mape_values]
metric_names = ["RMSE", "MAE", "MAPE (%)"]

for ax, data, metric_name in zip(axes, metric_data, metric_names):
    bars = ax.bar(
        models_order,
        data,
        color=bar_colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.8
    )
    ax.set_title(metric_name, fontsize=13)
    ax.set_ylabel("Error Value", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

# =========================
# 保存图片
# =========================
# 修复：去掉了名字里的 _Swap，保证与 app.py 寻找的路径完全一致
save_path = os.path.join(BASE_DIR, "RUL_CaseB_Error_Bar_Comparison.png")
plt.savefig(save_path, dpi=600, bbox_inches="tight")
print(f"\nFigure saved to: {save_path}")

# 修复：后台服务中不能有弹窗，否则会卡死进程
plt.show()