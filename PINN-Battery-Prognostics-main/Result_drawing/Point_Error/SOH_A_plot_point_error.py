import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Matplotlib 全局设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 结果文件目录
# =========================
BASE_DIR = r'D:\PycharmProjects\PINN-Battery-Prognostics-main\Results\4 Presentation\SoH Estimation'

results_paths = {
    "GRU": os.path.join(BASE_DIR, "SoH_CaseA_Baseline.pth"),
    "LSTM": os.path.join(BASE_DIR, "SoH_CaseA_LSTM_Baseline.pth"),
    "Verhulst-Sum": os.path.join(BASE_DIR, "SoH_CaseA_Verhulst_Sum.pth"),
    "Verhulst-AdpBal": os.path.join(BASE_DIR, "SoH_CaseA_Verhulst_AdpBal.pth")
}

# =========================
# 读取结果文件
# =========================
gru_results = torch.load(results_paths["GRU"], map_location='cpu', weights_only=False)
lstm_results = torch.load(results_paths["LSTM"], map_location='cpu', weights_only=False)
sum_results = torch.load(results_paths["Verhulst-Sum"], map_location='cpu', weights_only=False)
adpbal_results = torch.load(results_paths["Verhulst-AdpBal"], map_location='cpu', weights_only=False)

# =========================
# 提取真实值与预测值
# =========================
U_true = np.array(gru_results['U_true']).squeeze()

U_pred_gru = np.array(gru_results['U_pred']).squeeze()
U_pred_lstm = np.array(lstm_results['U_pred']).squeeze()
U_pred_sum = np.array(sum_results['U_pred']).squeeze()
U_pred_adpbal = np.array(adpbal_results['U_pred']).squeeze()

# 横轴优先使用 Cycles
if 'Cycles' in gru_results:
    x = np.array(gru_results['Cycles']).squeeze()
else:
    x = np.arange(len(U_true))

# =========================
# 长度保护
# =========================
min_len = min(
    len(x),
    len(U_true),
    len(U_pred_gru),
    len(U_pred_lstm),
    len(U_pred_sum),
    len(U_pred_adpbal)
)

x = x[:min_len]
U_true = U_true[:min_len]
U_pred_gru = U_pred_gru[:min_len]
U_pred_lstm = U_pred_lstm[:min_len]
U_pred_sum = U_pred_sum[:min_len]
U_pred_adpbal = U_pred_adpbal[:min_len]

# 按 cycles 排序
sort_idx = np.argsort(x)
x = x[sort_idx]
U_true = U_true[sort_idx]
U_pred_gru = U_pred_gru[sort_idx]
U_pred_lstm = U_pred_lstm[sort_idx]
U_pred_sum = U_pred_sum[sort_idx]
U_pred_adpbal = U_pred_adpbal[sort_idx]

# =========================
# 计算逐点绝对误差
# =========================
err_gru = np.abs(U_pred_gru - U_true)
err_lstm = np.abs(U_pred_lstm - U_true)
err_sum = np.abs(U_pred_sum - U_true)
err_adpbal = np.abs(U_pred_adpbal - U_true)

# =========================
# 创建对比图
# =========================
plt.figure(figsize=(10, 6))

plt.plot(x, err_gru, label='GRU', linestyle='--', color='#1f77b4', linewidth=1.8)
plt.plot(x, err_lstm, label='LSTM', linestyle='-.', color='#2ca02c', linewidth=1.8)
plt.plot(x, err_sum, label='PINN-Verhulst (Sum)', linestyle='--', color='#ff7f0e', linewidth=1.8)
plt.plot(x, err_adpbal, label='PINN-Verhulst (AdpBal)', linestyle='-', color='#d62728', linewidth=2.0)

# =========================
# 设置标签
# =========================
plt.xlabel('Monitoring Time (Unit: Cycles)')
plt.ylabel('Absolute Error')
# 不想放标题就删掉下一行
plt.title('Pointwise Absolute Error Comparison of SOH Estimation')

# 图例与网格
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# =========================
# 保存图片
# =========================
plt.savefig(
    os.path.join(BASE_DIR, 'SoH_CaseA_Pointwise_Absolute_Error_Comparison.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()