import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt

# 读取三个结果文件
baseline_results = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_Baseline.pth'), weights_only=False)
adpbal_results = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_DeepHPM_AdpBal.pth'), weights_only=False)
sum_results = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'RUL Prognostics', 'RUL_CaseA_DeepHPM_Sum.pth'), weights_only=False)

# 提取真实值与预测值
U_true = baseline_results['U_true']  # 真实值

# Baseline 模型的预测值
U_pred_baseline = baseline_results['U_pred']

# PINN-DeepHPM (AdpBal) 模型的预测值
U_pred_adpbal = adpbal_results['U_pred']

# PINN-DeepHPM (Sum) 模型的预测值
U_pred_sum = sum_results['U_pred']

# 创建对比图
plt.figure(figsize=(10, 6))

# 绘制真实值与三个预测值的对比
plt.plot(U_true, label='Ground Truth', linestyle='-', color='red')
plt.plot(U_pred_baseline, label='Baseline', linestyle='--', color='orange')
plt.plot(U_pred_sum, label='PINN-DeepHPM (Sum)', linestyle=':', color='orange')
plt.plot(U_pred_adpbal, label='PINN-DeepHPM (AdpBal)', linestyle='-', color='blue')

# 设置标签和标题
plt.xlabel('Monitoring Time (Unit: Cycles)')
plt.ylabel('Remaining Useful Life (Unit: Cycles)')
plt.title('Remaining Useful Life Prediction Comparison')

# 显示图例
plt.legend()

# 显示图表
plt.grid(True)
plt.show()