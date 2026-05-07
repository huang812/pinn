import os
import torch

# 定义一个函数来加载.pth文件并提取相关结果
def load_and_print_results(file_path, label):
    # 解决 PyTorch 2.6 兼容性问题：添加 weights_only=False
    results = torch.load(file_path, weights_only=False)

    # 提取结果，并确保它们是 Tensor 类型
    U_true = torch.tensor(results['U_true']) if not isinstance(results['U_true'], torch.Tensor) else results['U_true']
    U_pred = torch.tensor(results['U_pred']) if not isinstance(results['U_pred'], torch.Tensor) else results['U_pred']
    U_t_pred = torch.tensor(results['U_t_pred']) if not isinstance(results['U_t_pred'], torch.Tensor) else results['U_t_pred']
    cycles = results['Cycles']
    epochs = results['Epochs']

    # 计算训练误差的均方根误差（RMSE）
    RMSE_train = torch.sqrt(torch.mean((U_pred - U_true) ** 2))

    # 打印结果
    print(f"Results from {label}:")
    print(f"  - RMSE Train: {RMSE_train.item():.6f}")
    print(f"  - Epochs: {epochs[-1]}")
    print("-" * 50)


# =========================
# 路径自动获取逻辑
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
base_dir = os.path.join(project_root, "Results", "4 Presentation", "RUL Prognostics")

file_names = [
    "RUL_CaseA_Baseline.pth",
    "RUL_CaseA_DeepHPM_AdpBal.pth",
    "RUL_CaseA_DeepHPM_Sum.pth",
    "RUL_CaseB_Baseline.pth",
    "RUL_CaseB_DeepHPM_AdpBal.pth",
    "RUL_CaseB_DeepHPM_Sum.pth"
]

file_paths = [os.path.join(base_dir, name) for name in file_names]

for file_path in file_paths:
    label = os.path.basename(file_path).replace(".pth", "")
    if os.path.exists(file_path):
        load_and_print_results(file_path, label)
    else:
        print(f"[Error] File not found: {file_path}\n" + "-"*50)