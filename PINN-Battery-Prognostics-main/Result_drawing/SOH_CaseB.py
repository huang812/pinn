import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12  # 统一设置字体大小，后续可按需微调
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义颜色方案，让对比更鲜明
COLORS = {
    'Ground Truth': '#FF0000',  # 红色实线，真实值
    'Baseline': '#FF0000',  # 红色虚线
    'PINN-DeepHPM (AdpBal)': '#0000FF',  # 蓝色实线
    'PINN-DeepHPM (Sum)': '#0000FF',  # 蓝色点划线
    'PINN-Verhulst (AdpBal)': '#FFA500',  # 橙色实线
    'PINN-Verhulst (Sum)': '#FFA500'  # 橙色虚线
}

# 定义线条样式，与目标图匹配
LINESTYLES = {
    'Ground Truth': '-',  # 实线
    'Baseline': '--',  # 虚线
    'PINN-DeepHPM (AdpBal)': '-',  # 实线
    'PINN-DeepHPM (Sum)': '-.',  # 点划线
    'PINN-Verhulst (AdpBal)': '-',  # 实线
    'PINN-Verhulst (Sum)': '--'  # 虚线
}

# 结果文件所在的基础目录
# 请根据实际情况修改此路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results', '4 Presentation', 'SoH Estimation'))

# 构建完整文件路径，与目标方法名称对应
results_paths = {
    'Ground Truth': os.path.join(BASE_DIR, 'SoH_CaseB_Baseline.pth'),  # 假设Baseline对应真实值，根据实际调整
    'Baseline': os.path.join(BASE_DIR, 'SoH_CaseB_Baseline.pth'),
    'PINN-DeepHPM (AdpBal)': os.path.join(BASE_DIR, 'SoH_CaseB_DeepHPM_AdpBal.pth'),
    'PINN-DeepHPM (Sum)': os.path.join(BASE_DIR, 'SoH_CaseB_DeepHPM_Sum.pth'),
    'PINN-Verhulst (AdpBal)': os.path.join(BASE_DIR, 'SoH_CaseB_Verhulst_AdpBal.pth'),
    'PINN-Verhulst (Sum)': os.path.join(BASE_DIR, 'SoH_CaseB_Verhulst_Sum.pth')
}

# 检查文件是否存在
for model_name, path in results_paths.items():
    if not os.path.exists(path):
        print(f"Warning: File {path} does not exist!")
        print(f"Please check if the base directory is set correctly: {BASE_DIR}")
        print(f"Or modify the BASE_DIR variable in the code to the actual directory where the result files are stored.")
        exit(1)

# 创建图形（增大图幅，给RMSE信息预留空间，宽度和高度按需调整）
plt.figure(figsize=(12, 8))

# 提取 Ground Truth（假设Baseline文件里存的是真实值，根据实际数据调整）
ground_truth_results = torch.load(results_paths['Ground Truth'], weights_only=False)
cycles = ground_truth_results['Cycles']
ground_truth = ground_truth_results['U_true']

# 打印标题
print("=" * 50)
print("  RMSE Comparison of Battery State of Health Prediction Methods  ")
print("=" * 50)
print("Method Name\t\t\tRMSE Value")
print("-" * 50)

# 计算并存储所有模型的RMSE
rmse_values = {}

# 遍历所有模型结果
for model_name, path in results_paths.items():
    results = torch.load(path, weights_only=False)
    u_pred = results['U_pred']

    # 计算RMSE（与真实值ground_truth比较）
    rmse = np.sqrt(np.mean((u_pred - ground_truth) ** 2))
    rmse_values[model_name] = rmse

    # 绘制预测曲线，使用对应颜色和线条样式
    plt.plot(cycles, u_pred,
             label=model_name,
             color=COLORS[model_name],
             linestyle=LINESTYLES[model_name],
             linewidth=1.5)  # 加粗线条，增强对比

    # 打印RMSE结果（保留4位小数）
    print(f"{model_name}\t\t{rmse:.4f}")

# 找出最佳模型（RMSE最小）
best_model = min(rmse_values, key=rmse_values.get)
best_rmse = rmse_values[best_model]

# 打印最佳模型
print("-" * 50)
print(f"Best Model: {best_model}")
print(f"Lowest RMSE: {best_rmse:.4f}")
print("=" * 50)

# 设置图表属性，改为英文表述（与目标图一致）
plt.xlabel('Monitoring Time (Unit: Cycles)', fontsize=14)
plt.ylabel('State of Health (Unit: 1)', fontsize=14)
plt.title('Comparison of Battery State of Health Prediction Methods', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)  # 调整网格透明度，避免干扰曲线

# 调整图例位置，避免遮挡曲线，也可根据实际情况设为 'best' 等
plt.legend(fontsize=11, loc='upper right')

# 调整布局，给底端RMSE信息留空间
plt.tight_layout(rect=[0, 0.06, 1, 0.98])

# 构建RMSE显示文本
rmse_text = "RMSE: "
for model, rmse in rmse_values.items():
    rmse_text += f"{model}={rmse:.4f}, "
rmse_text = rmse_text.rstrip(", ")  # 去掉末尾多余的逗号和空格

# 将RMSE信息放置在图片底端居中，字体调小
plt.figtext(0.5, 0.01, rmse_text,
            ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

# 保存图表（高分辨率，可选），需显示完整可结合调整后的布局
# plt.savefig('SoH_Prediction_Comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()