% 假设 batch_combined 是一个包含所有电池数据的结构体数组
% Features 数组用来存储每个电池提取的特征
numBat = length(batch_combined);  % 计算电池的总数
Features = zeros(numBat, 8);  % 假设每个电池有 8 个特征

% 遍历每个电池并提取特征
for i = 1:numBat
    % 调用 ExtractFeatures 函数，传入第 i 个电池的数据
    Features(i, :) = ExtractFeatures(batch_combined(i));
end


function Features_i = ExtractFeatures(batch_combined_i)
    num_cycles = length(batch_combined_i.cycles);%初始化

    dQdV_max = zeros(num_cycles, 1);%初始化数组
    dQdV_min = zeros(num_cycles, 1);
    dQdV_var = zeros(num_cycles, 1);

    slope = zeros(num_cycles, 1);
    intercept = zeros(num_cycles, 1);

    for k = 1:num_cycles,
        %选择电池放电过程中的电压范围在2.7到3.3V之间的部分，用于计算 dQ/dV
        idx_dQdV = (batch_combined_i.Vdlin > 2.7) & (batch_combined_i.Vdlin < 3.3);

        len_Qdlin = length(batch_combined_i.cycles(k).Qdlin);
        len_Vdlin = length(batch_combined_i.Vdlin);
        if len_Qdlin ~= len_Vdlin
            error(sprintf('电池数据异常，循环 %d 中 Qdlin 长度: %d，Vdlin 长度: %d', k, len_Qdlin, len_Vdlin));
        end
        
        % 检查数据长度
        if length(batch_combined_i.cycles(k).Qdlin) ~= length(batch_combined_i.Vdlin)
            error('batch_combined_i.cycles(k).Qdlin 和 batch_combined_i.Vdlin 长度不匹配');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%% Kong et al. 2021 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        windowSize = 10;
        dV = -0.0015;
        thd = 1e-3;
        dQdV_tmp = diff(batch_combined_i.cycles(k).Qdlin)./diff(batch_combined_i.Vdlin); % dQdV_discharge has outliers
        dQdV_tmp_ma = tsmovavg(dQdV_tmp,'s', windowSize, 1);
        dQ_tmp_ma = dQdV_tmp_ma * dV;%对电池数据的平滑和计算 dQ/dV
        %找到 dQ/dV 曲线的峰值位置（idx_peak_dQ）
        %在峰值之后，找到放电电流变化量低于一个阈值 thd 的位置，并计算该区间的二次拟合结果。这个拟合的斜率和截距被存储在 slope(k) 和 intercept(k) 中
        N = length(dQ_tmp_ma);
        [~, idx_peak_dQ] = max(dQ_tmp_ma);
        for t = idx_peak_dQ:N,
            if dQ_tmp_ma(t) <= thd,
                idx_thd_dQ = t;
                y = dQ_tmp_ma(idx_peak_dQ:idx_thd_dQ);
                x = -batch_combined_i.cycles(k).Qdlin(idx_peak_dQ:idx_thd_dQ).^2;
                Y = y;
                X = [ones(idx_thd_dQ - idx_peak_dQ + 1, 1), x];
                p = pinv(X' * X) * X' * Y;
                slope(k) = p(1);
                intercept(k) = p(2);
                break;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %计算 dQ/dV 在指定电压范围内的最大值、最小值和方差
        dQdV_max(k) = max(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
        dQdV_min(k) = min(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
        dQdV_var(k) = var(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));

    end

    Features_i = [slope, intercept, batch_combined_i.summary.Tavg,...
        batch_combined_i.summary.IR, batch_combined_i.summary.chargetime,...
        dQdV_max, dQdV_min, dQdV_var];
end