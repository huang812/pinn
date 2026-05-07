import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
# Matplotlib 全局设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 读取结果文件
# =========================
baseline_results = torch.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_Baseline.pth'),
    map_location='cpu',
    weights_only=False
)

lstm_results = torch.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_LSTM_Baseline.pth'),
    map_location='cpu',
    weights_only=False
)

adpbal_results = torch.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_DeepHPM_AdpBal.pth'),
    map_location='cpu',
    weights_only=False
)

sum_results = torch.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_DeepHPM_Sum.pth'),
    map_location='cpu',
    weights_only=False
)

# =========================
# 提取真实值与预测值
# =========================
U_true = np.array(baseline_results['U_true']).squeeze()

U_pred_baseline = np.array(baseline_results['U_pred']).squeeze()
U_pred_lstm = np.array(lstm_results['U_pred']).squeeze()
U_pred_sum = np.array(sum_results['U_pred']).squeeze()
U_pred_adpbal = np.array(adpbal_results['U_pred']).squeeze()

# 横轴优先使用 Cycles，没有则使用样本序号
if 'Cycles' in baseline_results:
    x = np.array(baseline_results['Cycles']).squeeze()
else:
    x = np.arange(len(U_true))

# 长度保护，避免不同文件长度不一致
min_len = min(
    len(x),
    len(U_true),
    len(U_pred_baseline),
    len(U_pred_lstm),
    len(U_pred_sum),
    len(U_pred_adpbal)
)

x = x[:min_len]
U_true = U_true[:min_len]
U_pred_baseline = U_pred_baseline[:min_len]
U_pred_lstm = U_pred_lstm[:min_len]
U_pred_sum = U_pred_sum[:min_len]
U_pred_adpbal = U_pred_adpbal[:min_len]

# 如果横轴是 cycles，按 cycles 排序
sort_idx = np.argsort(x)
x = x[sort_idx]
U_true = U_true[sort_idx]
U_pred_baseline = U_pred_baseline[sort_idx]
U_pred_lstm = U_pred_lstm[sort_idx]
U_pred_sum = U_pred_sum[sort_idx]
U_pred_adpbal = U_pred_adpbal[sort_idx]

# =========================
# 创建对比图
# =========================
plt.figure(figsize=(10, 6))

plt.plot(x, U_true, label='Ground Truth', linestyle='-', color='black', linewidth=2.2)
plt.plot(x, U_pred_baseline, label='GRU', linestyle='--', color='#1f77b4', linewidth=1.8)
plt.plot(x, U_pred_lstm, label='LSTM', linestyle='-.', color='#2ca02c', linewidth=1.8)
plt.plot(x, U_pred_sum, label='PINN-DeepHPM (Sum)', linestyle='--', color='#ff7f0e', linewidth=1.8)
plt.plot(x, U_pred_adpbal, label='PINN-DeepHPM (AdpBal)', linestyle='-', color='#d62728', linewidth=2.0)

# =========================
# 设置标签
# =========================
plt.xlabel('Monitoring Time (Unit: Cycles)')
plt.ylabel('Remaining Useful Life (Unit: Cycles)')
plt.title('Remaining Useful Life Prediction Comparison')

# 图例与网格
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# =========================
# 保存图片 (这是缺失的关键步骤)
# =========================
# 动态定位到存放图片的结果文件夹
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics')
# 确保文件夹存在
os.makedirs(save_dir, exist_ok=True)
# 注意：这个文件名必须和我们在 app.py 里约定的一致
save_path = os.path.join(save_dir, "RUL_CaseA_Prediction_Comparison.png")

plt.savefig(save_path, dpi=600, bbox_inches="tight")
print(f"Figure saved to: {save_path}")

# 注意：一定要把 plt.show() 注释掉，否则后台静默运行时会卡住
plt.show()