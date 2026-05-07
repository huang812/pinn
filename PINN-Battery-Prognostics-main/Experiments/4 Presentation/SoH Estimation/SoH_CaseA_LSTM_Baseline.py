import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取实验设置
settings = torch.load('..\\..\\Settings\\settings_SoH_CaseA.pth')

seq_len = 1
perc_val = 0.2
num_rounds = 1
batch_size = settings['batch_size']
num_epoch = settings['num_epoch']

# 直接沿用原 settings 中的参数值，只是这里作为 LSTM 的层数和隐藏层维度使用
lstm_num_layers_list = settings['gru_num_layers']
lstm_hidden_size_list = settings['gru_hidden_size']

# 读取数据
addr = '..\\..\\..\\SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)

# 用于记录不同参数组合下 train/val/test 的均值与标准差
metric_mean = dict()
metric_std = dict()

metric_mean['train'] = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))
metric_mean['val'] = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))
metric_mean['test'] = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))

metric_std['train'] = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))
metric_std['val'] = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))
metric_std['test'] = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))

best_val_metric = np.inf
best_model = None
best_inputs_dict = None
best_targets_dict = None
best_results_epoch = None
best_lstm_num_layers = None
best_lstm_hidden_size = None

for l, lstm_num_layers in enumerate(lstm_num_layers_list):
    for n, lstm_hidden_size in enumerate(lstm_hidden_size_list):
        np.random.seed(1234)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)

        metric_rounds = dict()
        metric_rounds['train'] = np.zeros(num_rounds)
        metric_rounds['val'] = np.zeros(num_rounds)
        metric_rounds['test'] = np.zeros(num_rounds)

        for rd in range(num_rounds):
            # 选择训练/测试电池
            inputs_dict, targets_dict = func.create_chosen_cells(
                data,
                idx_cells_train=[91, 100],
                idx_cells_test=[124],
                perc_val=perc_val
            )

            inputs_train = inputs_dict['train'].to(device)
            inputs_val = inputs_dict['val'].to(device)
            inputs_test = inputs_dict['test'].to(device)

            # 仅取第 1 个目标维度（PCL），后续转为 SOH = 1 - PCL
            targets_train = targets_dict['train'][:, :, 0:1].to(device)
            targets_val = targets_dict['val'][:, :, 0:1].to(device)
            targets_test = targets_dict['test'][:, :, 0:1].to(device)

            inputs_dim = inputs_train.shape[2]
            outputs_dim = 1

            # 用训练集拟合标准化参数
            _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
            _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

            # 构建 DataLoader
            train_set = func.TensorDataset(inputs_train, targets_train)
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

            # LSTM 纯数据驱动模型
            model = func.DataDrivenLSTM(
                seq_len=seq_len,
                inputs_dim=inputs_dim,
                outputs_dim=outputs_dim,
                lstm_num_layers=lstm_num_layers,
                lstm_hidden_size=lstm_hidden_size,
                scaler_inputs=(mean_inputs_train, std_inputs_train),
                scaler_targets=(mean_targets_train, std_targets_train),
            ).to(device)

            # Baseline 模式下这三个量不参与优化，但为了接口统一保留
            log_sigma_u = torch.zeros((), device=device)
            log_sigma_f = torch.zeros((), device=device)
            log_sigma_f_t = torch.zeros((), device=device)

            criterion = func.My_loss(mode='Baseline')

            optimizer = optim.Adam(model.parameters(), lr=settings['lr'])
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=settings['step_size'],
                gamma=settings['gamma']
            )

            # 训练模型
            model, results_epoch = func.train(
                num_epoch=num_epoch,
                batch_size=batch_size,
                train_loader=train_loader,
                num_slices_train=inputs_train.shape[0],
                inputs_val=inputs_val,
                targets_val=targets_val,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                log_sigma_u=log_sigma_u,
                log_sigma_f=log_sigma_f,
                log_sigma_f_t=log_sigma_f_t
            )

            model.eval()

            # 训练集评估
            U_pred_train, _, _ = model(inputs=inputs_train)
            U_pred_train = 1.0 - U_pred_train
            targets_train_soh = 1.0 - targets_train
            RMSPE_train = torch.sqrt(
                torch.mean(((U_pred_train - targets_train_soh) / (targets_train_soh + 1e-8)) ** 2)
            )

            # 验证集评估
            U_pred_val, _, _ = model(inputs=inputs_val)
            U_pred_val = 1.0 - U_pred_val
            targets_val_soh = 1.0 - targets_val
            RMSPE_val = torch.sqrt(
                torch.mean(((U_pred_val - targets_val_soh) / (targets_val_soh + 1e-8)) ** 2)
            )

            # 测试集评估
            U_pred_test, _, _ = model(inputs=inputs_test)
            U_pred_test = 1.0 - U_pred_test
            targets_test_soh = 1.0 - targets_test
            RMSPE_test = torch.sqrt(
                torch.mean(((U_pred_test - targets_test_soh) / (targets_test_soh + 1e-8)) ** 2)
            )

            metric_rounds['train'][rd] = RMSPE_train.detach().cpu().numpy()
            metric_rounds['val'][rd] = RMSPE_val.detach().cpu().numpy()
            metric_rounds['test'][rd] = RMSPE_test.detach().cpu().numpy()

            # 保存当前最优模型（按验证集最优）
            if metric_rounds['val'][rd] < best_val_metric:
                best_val_metric = metric_rounds['val'][rd]
                best_model = model
                best_inputs_dict = inputs_dict
                best_targets_dict = targets_dict
                best_results_epoch = results_epoch
                best_lstm_num_layers = lstm_num_layers
                best_lstm_hidden_size = lstm_hidden_size

        # 统计均值和标准差
        metric_mean['train'][l, n] = np.mean(metric_rounds['train'])
        metric_mean['val'][l, n] = np.mean(metric_rounds['val'])
        metric_mean['test'][l, n] = np.mean(metric_rounds['test'])

        metric_std['train'][l, n] = np.std(metric_rounds['train'])
        metric_std['val'][l, n] = np.std(metric_rounds['val'])
        metric_std['test'][l, n] = np.std(metric_rounds['test'])

        print('=' * 80)
        print(f'LSTM layers: {lstm_num_layers}, hidden size: {lstm_hidden_size}')
        print(f"Train RMSPE mean: {metric_mean['train'][l, n]:.6f}, std: {metric_std['train'][l, n]:.6f}")
        print(f"Val   RMSPE mean: {metric_mean['val'][l, n]:.6f}, std: {metric_std['val'][l, n]:.6f}")
        print(f"Test  RMSPE mean: {metric_mean['test'][l, n]:.6f}, std: {metric_std['test'][l, n]:.6f}")
        print('=' * 80)

# 用最优模型在测试集上输出结果
best_model.eval()
inputs_test = best_inputs_dict['test'].to(device)
targets_test = best_targets_dict['test'][:, :, 0:1].to(device)

U_pred_test, _, _ = best_model(inputs=inputs_test)
U_pred_test = 1.0 - U_pred_test
targets_test = 1.0 - targets_test

results = dict()
results['U_true'] = targets_test.detach().cpu().numpy().squeeze()
results['U_pred'] = U_pred_test.detach().cpu().numpy().squeeze()
results['U_t_pred'] = best_model.U_t.detach().cpu().numpy().squeeze()
results['Cycles'] = inputs_test[:, :, -1:].detach().cpu().numpy().squeeze()
results['Epochs'] = np.arange(0, num_epoch)

results['metric_mean'] = metric_mean
results['metric_std'] = metric_std
results['best_val_metric'] = best_val_metric
results['best_lstm_num_layers'] = best_lstm_num_layers
results['best_lstm_hidden_size'] = best_lstm_hidden_size
results['loss_train'] = best_results_epoch['loss_train'].detach().cpu().numpy()
results['loss_val'] = best_results_epoch['loss_val'].detach().cpu().numpy()

torch.save(
    results,
    '..\\..\\..\\Results\\4 Presentation\\SoH Estimation\\SoH_CaseA_LSTM_Baseline.pth'
)

print('\nBest model summary:')
print(f'Best validation RMSPE: {best_val_metric:.6f}')
print(f'Best LSTM layers: {best_lstm_num_layers}')
print(f'Best LSTM hidden size: {best_lstm_hidden_size}')
print('Results saved to: ..\\..\\..\\Results\\4 Presentation\\SoH Estimation\\SoH_CaseA_LSTM_Baseline.pth')