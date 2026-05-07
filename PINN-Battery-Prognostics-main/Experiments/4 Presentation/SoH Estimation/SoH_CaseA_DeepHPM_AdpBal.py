import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = torch.load('..\\..\\Settings\\settings_SoH_CaseA.pth')#加载SOHcaseA配置文件，里面包含模型训练的超参数
seq_len = 1
perc_val = 0.2
num_rounds = 1
batch_size = settings['batch_size']
num_epoch = settings['num_epoch']
# 修改为GRU相关参数
gru_num_layers = settings['gru_num_layers']
gru_hidden_size = settings['gru_hidden_size']
inputs_lib_dynamical = settings['inputs_lib_dynamical']
inputs_dim_lib_dynamical = settings['inputs_dim_lib_dynamical']

addr = '..\\..\\..\\SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)#加载电池数据集，seq_len设定为1，表示数据的时间序列长度

metric_mean = dict()#初始化字典，用来存储每轮训练、验证和测试集上的平均值和标准差
metric_std = dict()
metric_mean['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['test'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['test'] = np.zeros((len(inputs_lib_dynamical), 1))#使用np.zeros()初始化，长度为inputs_lib_dynamical的长度，即动态输入的数量

for l in range(len(inputs_lib_dynamical)):
    inputs_dynamical, inputs_dim_dynamical = inputs_lib_dynamical[l], inputs_dim_lib_dynamical[l]
    np.random.seed(1234)
    torch.manual_seed(1234)
    metric_rounds = dict()
    metric_rounds['train'] = np.zeros(num_rounds)
    metric_rounds['val'] = np.zeros(num_rounds)
    metric_rounds['test'] = np.zeros(num_rounds)#对于每个动态输入，设置网络层数并初始化随机种子
    for round in range(num_rounds):
        inputs_dict, targets_dict = func.create_chosen_cells(
            data,
            idx_cells_train=[91, 100],
            idx_cells_test=[124],
            perc_val=perc_val
        )#获取训练、验证和测试数据
        inputs_train = inputs_dict['train'].to(device)
        inputs_val = inputs_dict['val'].to(device)
        inputs_test = inputs_dict['test'].to(device)
        targets_train = targets_dict['train'][:, :, 0:1].to(device)
        targets_val = targets_dict['val'][:, :, 0:1].to(device)
        targets_test = targets_dict['test'][:, :, 0:1].to(device)#to device将数据传输到指定的设备gpu或cpu

        inputs_dim = inputs_train.shape[2]
        outputs_dim = 1

        _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
        _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')#数据标准化，得到均值和标准差。

        train_set = func.TensorDataset(inputs_train, targets_train)  # J_train is a placeholder
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )#创建训练数据的批次

        model = func.DeepHPMNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            # 修改为GRU相关参数
            gru_num_layers=gru_num_layers,
            gru_hidden_size=gru_hidden_size,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
            inputs_dynamical=inputs_dynamical,
            inputs_dim_dynamical=inputs_dim_dynamical
        ).to(device)#定义模型
#torch.randn() 用于生成一个服从标准正态分布（均值为0，标准差为1）的随机数张量。
        log_sigma_u = torch.randn((), requires_grad=True)
        log_sigma_f = torch.randn((), requires_grad=True)
        log_sigma_f_t = torch.randn((), requires_grad=True)

        criterion = func.My_loss(mode='AdpBal')#自定义的损失函数，模式为AdpBal。

        params = ([p for p in model.parameters()] + [log_sigma_u] + [log_sigma_f] + [log_sigma_f_t])
        optimizer = optim.Adam(params, lr=settings['lr'])
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

        U_pred_train, F_pred_train, _ = model(inputs=inputs_train)
        U_pred_train = 1. - U_pred_train
        targets_train = 1. - targets_train
        RMSPE_train = torch.sqrt(torch.mean(((U_pred_train - targets_train) / targets_train) ** 2))#训练，并计算RMSPE

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

    metric_mean['train'][l] = np.mean(metric_rounds['train'])
    metric_mean['val'][l] = np.mean(metric_rounds['val'])
    metric_mean['test'][l] = np.mean(metric_rounds['test'])
    metric_std['train'][l] = np.std(metric_rounds['train'])
    metric_std['val'][l] = np.std(metric_rounds['val'])
    metric_std['test'][l] = np.std(metric_rounds['test'])#存储结果

model.eval()#将模型切换到评估模式（关闭Dropout和BatchNorm）
inputs_test = inputs_dict['test'].to(device)
targets_test = targets_dict['test'][:, :, 0:1].to(device)#提取测试集的输入数据和目标数据，将其传输到GPU或CPU上
U_pred_test, F_pred_test, _ = model(inputs=inputs_test)#使用测试集数据输入模型，得到模型的预测结果，其中U是模型预测的输出，F是预测的辅助变量（可能是梯度或其他相关的输出）。
U_pred_test = 1. - U_pred_test
targets_test = 1. - targets_test#将预测结果和目标结果进行处理，将它们转换为[0, 1]之间的值

results = dict()
results['U_true'] = targets_test.detach().cpu().numpy().squeeze()#目标的真实值
results['U_pred'] = U_pred_test.detach().cpu().numpy().squeeze()#模型预测值
results['U_t_pred'] = model.U_t.detach().cpu().numpy().squeeze()#模型预测的时间梯度
results['Cycles'] = inputs_test[:, :, -1:].detach().cpu().numpy().squeeze()#测试数据的周期信息（从inputs_test的最后一个维度提取）
results['Epochs'] = np.arange(0, num_epoch)#训练过程的每个周期
results['lambda_U'] = results_epoch['var_U']
results['lambda_F'] = results_epoch['var_F']#在每个周期的训练过程中模型的输出不确定度（方差）
torch.save(results, '..\\..\\..\\Results\\4 Presentation\\SoH Estimation\\SoH_CaseA_DeepHPM_AdpBal.pth')#存储数据
pass
