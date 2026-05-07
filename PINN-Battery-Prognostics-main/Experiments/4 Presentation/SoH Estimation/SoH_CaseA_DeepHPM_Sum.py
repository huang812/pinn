import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = torch.load('..\\..\\Settings\\settings_SoH_CaseA.pth')
seq_len = 1
perc_val = 0.2
num_rounds = 1
batch_size = settings['batch_size']#从配置文件中读取参数
num_epoch = settings['num_epoch']
gru_num_layers = settings['gru_num_layers']  # 添加GRU层数
gru_hidden_size = settings['gru_hidden_size']  # 添加GRU隐藏层大小
inputs_lib_dynamical = settings['inputs_lib_dynamical']
inputs_dim_lib_dynamical = settings['inputs_dim_lib_dynamical']

addr = '..\\..\\..\\SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)

metric_mean = dict()
metric_std = dict()#metric_mean 和 metric_std 字典用于存储每个数据集（训练集、验证集、测试集）的均值和标准差。这些字典分别初始化为零矩阵，矩阵的行数等于inputs_lib_dynamical的长度。
metric_mean['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['test'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['test'] = np.zeros((len(inputs_lib_dynamical), 1))

for l in range(len(inputs_lib_dynamical)):
    inputs_dynamical, inputs_dim_dynamical = inputs_lib_dynamical[l], inputs_dim_lib_dynamical[l]

    np.random.seed(1234)
    torch.manual_seed(1234)
    metric_rounds = dict()#记录每一轮训练、验证和测试集的指标
    metric_rounds['train'] = np.zeros(num_rounds)
    metric_rounds['val'] = np.zeros(num_rounds)
    metric_rounds['test'] = np.zeros(num_rounds)

    for round in range(num_rounds):
        inputs_dict, targets_dict = func.create_chosen_cells(
            data,
            idx_cells_train=[91, 100],
            idx_cells_test=[124],
            perc_val=perc_val
        )
        inputs_train = inputs_dict['train'].to(device)
        inputs_val = inputs_dict['val'].to(device)
        inputs_test = inputs_dict['test'].to(device)
        targets_train = targets_dict['train'][:, :, 0:1].to(device)
        targets_val = targets_dict['val'][:, :, 0:1].to(device)
        targets_test = targets_dict['test'][:, :, 0:1].to(device)

        inputs_dim = inputs_train.shape[2]
        outputs_dim = 1

        _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
        _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')#将输入数据和目标数据的均值调整为0，标准差调整为1

        train_set = func.TensorDataset(inputs_train, targets_train)  # J_train is a placeholder#使用func.TensorDataset将训练集的输入数据和目标数据封装在一起，生成一个 PyTorch 数据集（TensorDataset）
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,#每个epoch开始时将训练数据打乱，以避免模型对数据顺序的依赖
            num_workers=0,
            drop_last=True#如果数据集的大小不能整除批次大小，丢弃最后一个不完整的批次
        )

        model = func.DeepHPMNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            gru_num_layers=gru_num_layers,  # GRU层数
            gru_hidden_size=gru_hidden_size,  # GRU隐藏层大小
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
            inputs_dynamical=inputs_dynamical,
            inputs_dim_dynamical=inputs_dim_dynamical
        ).to(device)#传入模型参数

        log_sigma_u = torch.zeros(())#创建三个零维张量，用于存储log_sigma变量
        log_sigma_f = torch.zeros(())
        log_sigma_f_t = torch.zeros(())

        criterion = func.My_loss(mode='Sum')

        params = ([p for p in model.parameters()])#获取模型的所有参数，以便优化器进行更新
        optimizer = optim.Adam(params, lr=settings['lr'])#使用Adam优化器进行参数优化，并设置学习率
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings['step_size'], gamma=settings['gamma'])#使用学习率调度器，按照设定的step_size每隔一定步数衰减学习率，衰减比例为gamma。
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
        )#训练模型

        model.eval()#评估模型
#使用模型进行预测，并计算RMSPE。通过1 - U_pred和1 - targets进行变换，可能是因为目标和预测值是经过归一化的？
        U_pred_train, F_pred_train, _ = model(inputs=inputs_train)
        U_pred_train = 1. - U_pred_train
        targets_train = 1. - targets_train
        RMSPE_train = torch.sqrt(torch.mean(((U_pred_train - targets_train) / targets_train) ** 2))

        U_pred_val, F_pred_val, _ = model(inputs=inputs_val)
        U_pred_val = 1. - U_pred_val
        targets_val = 1. - targets_val
        RMSPE_val = torch.sqrt(torch.mean(((U_pred_val - targets_val) / targets_val) ** 2))

        U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
        U_pred_test = 1. - U_pred_test
        targets_test = 1. - targets_test
        RMSPE_test = torch.sqrt(torch.mean(((U_pred_test - targets_test) / targets_test) ** 2))

        metric_rounds['train'][round] = RMSPE_train.detach().cpu().numpy()
        metric_rounds['val'][round] = RMSPE_val.detach().cpu().numpy()
        metric_rounds['test'][round] = RMSPE_test.detach().cpu().numpy()
#计算并存储每个数据集（训练集、验证集、测试集）的 RMSPE 的平均值和标准差。
    metric_mean['train'][l] = np.mean(metric_rounds['train'])
    metric_mean['val'][l] = np.mean(metric_rounds['val'])
    metric_mean['test'][l] = np.mean(metric_rounds['test'])
    metric_std['train'][l] = np.std(metric_rounds['train'])
    metric_std['val'][l] = np.std(metric_rounds['val'])
    metric_std['test'][l] = np.std(metric_rounds['test'])

model.eval()
inputs_test = inputs_dict['test'].to(device)
targets_test = targets_dict['test'][:, :, 0:1].to(device)
U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
U_pred_test = 1. - U_pred_test
targets_test = 1. - targets_test

results = dict()
results['U_true'] = targets_test.detach().cpu().numpy().squeeze()
results['U_pred'] = U_pred_test.detach().cpu().numpy().squeeze()
results['U_t_pred'] = model.U_t.detach().cpu().numpy().squeeze()
results['Cycles'] = inputs_test[:, :, -1:].detach().cpu().numpy().squeeze()
results['Epochs'] = np.arange(0, num_epoch)
torch.save(results, '..\\..\\..\\Results\\4 Presentation\\SoH Estimation\\SoH_CaseA_DeepHPM_Sum.pth')
pass
