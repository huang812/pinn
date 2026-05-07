%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to show an example of loading the dataset associated with      %
% K.A. Severson, P.M. Attia et al. Data Driven Prediction of Battery    %
% Cycle Life Before Capacity Degradation (2019) Nature Energy           %
% This code assumes the data is in a subdirectory named 'Data'          %
% The structs include information on the 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc

load('2017-05-12_batchdata_updated_struct_errorcorrect')

batch1 = batch; 
numBat1 = size(batch1,2);%%加载的数据被存储到batch1中，并计算该批次电池的数量

load('2017-06-30_batchdata_updated_struct_errorcorrect') %%加载第二批数据

%Some batteries continued from the first run into the second. We append 
%those to the first batch before continuing.
%My notes: The add_len are added to the first batch continue to that batch
add_len = [661, 980, 1059, 207, 481];%%记录了每个电池从第一批数据延续到第二批数据时的循环数
summary_var_list = {'cycle','QDischarge','QCharge','IR','Tmax','Tavg',...
    'Tmin','chargetime'};
batch2_idx = [8:10,16:17];%%确定了需要从第二批数据中合并的电池
for i=1:5%%将数据附加到batch1中
    batch1(i).cycles(end+1:end+add_len(i)+1) = batch(batch2_idx(i)).cycles;
    batch1(i).cycle_life = batch1(i).cycle_life + add_len(i) + 1;
    batch1(i).summary.cycle(end+1:end+add_len(i)+1) = ...
        batch1(i).summary.cycle(end) + batch(batch2_idx(i)).summary.cycle;
    batch1(i).summary.QDischarge(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.QDischarge;
    batch1(i).summary.QCharge(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.QCharge;
    batch1(i).summary.IR(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.IR;
    batch1(i).summary.Tmax(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tmax;
    batch1(i).summary.Tavg(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tavg;
    batch1(i).summary.Tmin(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tmin;
    batch1(i).summary.chargetime(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.chargetime;
end%%批次一中的电池就合并了第二批的数据。随后，原始的第二批数据（batch([8:10,16:17])）被移除，剩下的被保存为batch2。

batch([8:10,16:17]) = [];
batch2 = batch;
numBat2 = size(batch2,2);
clearvars batch

load('2018-04-12_batchdata_updated_struct_errorcorrect')%%加载第三批数据
batch3 = batch;
batch3(38) = []; %移除错误数据remove channel 46 upfront; there was a problem with 
%the data collection for this channel
numBat3 = size(batch3,2);
endcap3 = zeros(numBat3,1);
clearvars batch
for i = 1:numBat3
    endcap3(i) = batch3(i).summary.QDischarge(end);
end
rind = find(endcap3 > 0.885);
batch3(rind) = [];%%从第三批数据中移除放电容量（QDischarge）大于0.885的电池

%remove the noisy Batch 8 batteries移除有噪声的数据
nind = [3, 40:41];
batch3(nind) = [];
numBat3 = size(batch3,2);

batch_combined = [batch1, batch2, batch3];%%将一二三批的数据合并在一起
numBat = numBat1 + numBat2 + numBat3;

%optionally remove the batteries that do not finish in Batch 1; depending
%on the modeling goal, you may not want to do this step
batch_combined([9,11,13,14,23]) = [];
numBat = numBat - 5;
numBat1 = numBat1 - 5; 

clearvars -except batch_combined numBat1 numBat2 numBat3 numBat

%% Output variable
%Extract the number of cycles to 0.88; this is the output variable used in
%modeling for the paper

bat_label = zeros(numBat,1);%%提取每个电池的循环次数，直到其放电容量降到0.88以下
%%这个值是该论文中用于建模的输出变量
for i = 1:numBat
    if batch_combined(i).summary.QDischarge(end) < 0.88
        bat_label(i) = find(batch_combined(i).summary.QDischarge < 0.88,1);
        
    else
        bat_label(i) = size(batch_combined(i).cycles,2) + 1;
    end
end%%判断每个电池的放电容量，若低于0.88，就记录下该电池循环次数，否则记录下最大循环次数加1

figure()%%可视化每个电池放电容量随时间的变化
hold on
for i = 1:numBat
    plot(batch_combined(i).summary.QDischarge,'.-')
end

%% Train and Test Split
% If you are interested in using the same train/test split as the paper,
% use the indices specified below

test_ind = [1:2:(numBat1+numBat2),84];%%划分训练集和测试集
train_ind = 1:(numBat1+numBat2);
train_ind(test_ind) = [];
secondary_test_ind = numBat-numBat3+1:numBat;

all_features = {}; 
for i = 1:length(batch_combined)
    disp(batch_combined(i));
    required_fields = {'cycles', 'Vdlin','summary'}; 
    sub_required_fields = {'Qdlin', 'discharge_dQdV'}; 
    summary_required_fields = {'Tavg', 'IR', 'chargetime'}; 

    % 检查顶层字段
    for j = 1:length(required_fields)
        field_name = required_fields{j};
        if ~isfield(batch_combined(i), field_name)
            fprintf('电池 %d 缺少字段 %s\n', i, field_name);
            continue; 
        elseif strcmp(field_name, 'cycles')
            % 检查 cycles 结构体中的子字段
            for k = 1:length(sub_required_fields)
                sub_field_name = sub_required_fields{k};
                if ~isfield(batch_combined(i).cycles, sub_field_name)
                    fprintf('电池 %d 的 cycles 字段中缺少子字段 %s\n', i, sub_field_name);
                    continue; 
                end
            end
        elseif strcmp(field_name,'summary')
            % 检查 summary 结构体中的子字段
            for k = 1:length(summary_required_fields)
                sub_field_name = summary_required_fields{k};
                if ~isfield(batch_combined(i).summary, sub_field_name)
                    fprintf('电池 %d 的 summary 字段中缺少子字段 %s\n', i, sub_field_name);
                    continue; 
                end
            end
        end
    end
    % 新增查看数据长度代码
    num_cycles = length(batch_combined(i).cycles);
    for k = 1:num_cycles
        len_Qdlin = length(batch_combined(i).cycles(k).Qdlin);
        len_Vdlin = length(batch_combined(i).Vdlin);
        fprintf('电池 %d，循环 %d 中，Qdlin 长度: %d，Vdlin 长度: %d\n', i, k, len_Qdlin, len_Vdlin);
    end
    Features_i = ExtractFeatures(batch_combined(i));
    all_features{end + 1} = Features_i; 
end
