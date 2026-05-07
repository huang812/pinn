import torch
import matplotlib.pyplot as plt
import numpy as np
results = torch.load('..\\..\\..\\Results\\4 Presentation\\SoH Estimation\\SoH_CaseA_DeepHPM_AdpBal.pth')
# 假设这些数据已经从 .pth 文件中加载
# 这里的数据是示例，你可以根据自己的实际情况替换它们
U_true = results['U_true']  # 真实值
U_pred = results['U_pred']  # 预测值
#lambda_U = results['lambda_U']  # 预测的不确定度（方差）

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 绘制真实值和预测值
plt.subplot(2, 1, 1)
plt.plot(U_true, label='True Values', color='blue', linewidth=2)
plt.plot(U_pred, label='Predicted Values', color='red', linestyle='--', linewidth=2)
plt.title('True vs Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('State of Health (SoH)')
plt.legend()

# 绘制预测的方差
#plt.subplot(2, 1, 2)
#plt.plot(lambda_U, label='Prediction Uncertainty (Lambda_U)', color='green', linewidth=2)
#plt.title('Prediction Uncertainty (Lambda_U)')
#plt.xlabel('Time Steps')
#plt.ylabel('Uncertainty')
#plt.legend()

plt.tight_layout()
plt.show()