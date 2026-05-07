required_fields = {'cycles', 'Vdlin','summary'}; % 先列出 ExtractFeatures 函数依赖的主要顶层字段
sub_required_fields = {'Qdlin', 'discharge_dQdV'}; % cycles 结构体中可能依赖的子字段
summary_required_fields = {'Tavg', 'IR', 'chargetime'}; % summary 结构体中可能依赖的子字段

numBat = length(batch_combined);
for i = 1:numBat
    % 检查顶层字段
    for j = 1:length(required_fields)
        field_name = required_fields{j};
        if ~isfield(batch_combined(i), field_name)
            fprintf('电池 %d 缺少字段 %s\n', i, field_name);
        elseif strcmp(field_name, 'cycles')
            % 检查 cycles 结构体中的子字段
            for k = 1:length(sub_required_fields)
                sub_field_name = sub_required_fields{k};
                if ~isfield(batch_combined(i).cycles, sub_field_name)
                    fprintf('电池 %d 的 cycles 字段中缺少子字段 %s\n', i, sub_field_name);
                end
            end
        elseif strcmp(field_name,'summary')
            % 检查 summary 结构体中的子字段
            for k = 1:length(summary_required_fields)
                sub_field_name = summary_required_fields{k};
                if ~isfield(batch_combined(i).summary, sub_field_name)
                    fprintf('电池 %d 的 summary 字段中缺少子字段 %s\n', i, sub_field_name);
                end
            end
        end
    end
end