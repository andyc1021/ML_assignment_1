function [engine_labled_data, service_idx] = process_engine_data(engine_raw_data_train)
% Aircraft Engine

%% Find last cycle index
max_cycle_idx = zeros(92,1);
cycle_idx = 1;
for i = 1:height(engine_raw_data_train)
    if i > 1 && table2array(engine_raw_data_train(i,1)) ~= table2array(engine_raw_data_train(i-1,1))
        max_cycle_idx(cycle_idx) = i-1;
        cycle_idx = cycle_idx + 1;
    end
end
max_cycle_idx(end) = height(engine_raw_data_train);

%% Find almost ready for service interval (<5 of max cycle index)
service_idx = [];
for j = 1:length(max_cycle_idx)
    add_idx = [max_cycle_idx(j) - 2, max_cycle_idx(j) - 1, max_cycle_idx(j)];%[max_cycle_idx(j) - 4, max_cycle_idx(j) - 3, max_cycle_idx(j) - 2, max_cycle_idx(j) - 1, max_cycle_idx(j)];
    service_idx = [service_idx, add_idx];
end
service_idx = service_idx';

%% Add new categorical string to table
need_service_str = strings(height(engine_raw_data_train),1);
for i = 1:height(engine_raw_data_train)
    if ismember(i, service_idx)
        need_service_str(i) = "NeedService";
    else
        need_service_str(i) = "OK";
    end
end
need_service_cat = categorical(need_service_str);

%%
engine_labled_data = engine_raw_data_train;
engine_labled_data.NeedService = need_service_cat;
end