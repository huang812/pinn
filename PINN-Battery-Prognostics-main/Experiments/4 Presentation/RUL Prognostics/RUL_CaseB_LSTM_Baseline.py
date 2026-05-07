import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取实验设置
settings = torch.load('..\\..\\Settings\\settings_RUL_CaseB.pth')
seq_len = 1
perc_val = 0.2
num_rounds = 1
batch_size = settings['batch_size']
num_epoch = settings['num_epoch']

lstm_num_layers_list = settings['gru_num_layers']  # LSTM层数
lstm_hidden_size_list = settings['gru_hidden_size']  # LSTM隐藏层维度

addr = '..\\..\\..\\SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)

metric_mean = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))
metric_std = np.zeros((len(lstm_num_layers_list), len(lstm_hidden_size_list)))

for l, lstm_num_layers in enumerate(lstm_num_layers_list):
    for n, lstm_hidden_size in enumerate(lstm_hidden_size_list):
        np.random.seed(1234)
        torch.manual_seed(1234)

        metric_rounds = {'train': np.zeros(num_rounds), 'val': np.zeros(num_rounds), 'test': np.zeros(num_rounds)}

        for rd in range(num_rounds):
            # 选取训练/测试电池
            inputs_dict, targets_dict = func.create_chosen_cells(
                data,
                idx_cells_train=[101, 108, 120],
                idx_cells_test=[116],
                perc_val=perc_val
            )
            inputs_train = inputs_dict['train'].to(device)
            inputs_val = inputs_dict['val'].to(device)
            inputs_test = inputs_dict['test'].to(device)
            targets_train = targets_dict['train'][:, :, 1:].to(device)
            targets_val = targets_dict['val'][:, :, 1:].to(device)
            targets_test = targets_dict['test'][:, :, 1:].to(device)

            inputs_dim = inputs_train.shape[2]
            outputs_dim = 1

            _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
            _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

            train_set = func.TensorDataset(inputs_train, targets_train)
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

            # LSTM模型
            model = func.DataDrivenLSTM(
                seq_len=seq_len,
                inputs_dim=inputs_dim,
                outputs_dim=outputs_dim,
                lstm_num_layers=lstm_num_layers,
                lstm_hidden_size=lstm_hidden_size,
                scaler_inputs=(mean_inputs_train, std_inputs_train),
                scaler_targets=(mean_targets_train, std_targets_train)
            ).to(device)

            log_sigma_u = torch.zeros(())
            log_sigma_f = torch.zeros(())
            log_sigma_f_t = torch.zeros(())

            criterion = func.My_loss(mode='Baseline')
            optimizer = optim.Adam(model.parameters(), lr=settings['lr'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings['step_size'], gamma=settings['gamma'])

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

            # 训练/验证/测试指标
            U_pred_train, _, _ = model(inputs=inputs_train)
            RMSE_train = torch.sqrt(torch.mean((U_pred_train - targets_train) ** 2))
            U_pred_val, _, _ = model(inputs=inputs_val)
            RMSE_val = torch.sqrt(torch.mean((U_pred_val - targets_val) ** 2))
            U_pred_test, _, _ = model(inputs=inputs_test)
            RMSE_test = torch.sqrt(torch.mean((U_pred_test - targets_test) ** 2))

            metric_rounds['train'][rd] = RMSE_train.detach().cpu().numpy()
            metric_rounds['val'][rd] = RMSE_val.detach().cpu().numpy()
            metric_rounds['test'][rd] = RMSE_test.detach().cpu().numpy()

        metric_mean[l, n] = np.mean(metric_rounds['test'])
        metric_std[l, n] = np.std(metric_rounds['test'])

# 保存最优模型在测试集上的预测
model.eval()
inputs_test = inputs_dict['test'].to(device)
targets_test = targets_dict['test'][:, :, 1:].to(device)
U_pred_test, _, _ = model(inputs=inputs_test)

results = {
    'U_true': targets_test.detach().cpu().numpy().squeeze(),
    'U_pred': U_pred_test.detach().cpu().numpy().squeeze(),
    'U_t_pred': model.U_t.detach().cpu().numpy().squeeze(),
    'Cycles': inputs_test[:, :, -1:].detach().cpu().numpy().squeeze(),
    'Epochs': np.arange(0, num_epoch),
    'metric_mean': metric_mean,
    'metric_std': metric_std
}

torch.save(results, '..\\..\\..\\Results\\4 Presentation\\RUL Prognostics\\RUL_CaseB_LSTM_Baseline.pth')