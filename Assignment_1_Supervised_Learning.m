%%
% Author: Andy Cheng
% Date: 2/1/2024
% Class: CS 7641 Machine Learning
% Assignment: A1 Supervised Learning
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data
% Apple Quality
apple_raw_data = readtable('apple_quality.csv');

% Aircraft Engine
engine_raw_data_train = readtable('PM_train.xlsx');
[engine_labled_data, engine_service_idx] = process_engine_data(engine_raw_data_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preprocess data
% Set rng constant
rng(123456);
tallrng(123456);

% Apple Quality
apple_input = apple_raw_data(1:4000, 2:8);
apple_label = apple_raw_data(1:4000, 9);
apple_label_categ = categorical(table2array(apple_label));

% Random select to split train/test sets
apple_train_set_n = ceil(0.85*length(apple_label_categ));
apple_test_set_n = floor(0.15*length(apple_label_categ));
apple_indices = randperm(length(apple_label_categ));
apple_train_indices = apple_indices(1:apple_train_set_n);
apple_test_indices = apple_indices(apple_train_set_n + 1:end);

apple_train_set = apple_input(apple_train_indices, :);
apple_train_set_cat = apple_label_categ(apple_train_indices);

apple_test_set = apple_input(apple_test_indices, :);
apple_test_set_cat = apple_label_categ(apple_test_indices);

%% Aircraft Engine
% Need to get equal 50-50 split so create new data set
engine_OK_set = engine_labled_data(setdiff(1:end,engine_service_idx), :);
engine_service_set = engine_labled_data(engine_service_idx, :);
engine_OK_indices = randperm(height(engine_OK_set));
engine_OK_indices = engine_OK_indices(1:height(engine_service_set));
engine_OK_set_reduced = engine_OK_set(engine_OK_indices, :);

% Combine 50-50 OK-Service split tables and randomize it.
engine_combined_table = [engine_service_set; engine_OK_set_reduced];
engine_rand_idx = randperm(height(engine_combined_table));
engine_rand_idx_2 = randperm(height(engine_combined_table));
engine_combined_table = engine_combined_table(engine_rand_idx, :);
engine_combined_table = engine_combined_table(engine_rand_idx_2, :);

% Randomly select into 70-30 train-test sets
%good_engine_idx = [5:7,9:12,14:18,20,23:26]; % for states that matter
engine_input = engine_combined_table(:, 3:26);
engine_label = engine_combined_table(:, 27);
engine_label_categ = categorical(table2array(engine_label));

engine_train_set_n = ceil(0.85*length(engine_label_categ));
engine_test_set_n = floor(0.15*length(engine_label_categ));
engine_indices = randperm(length(engine_label_categ));
engine_train_indices = engine_indices(1:engine_train_set_n);
engine_test_indices = engine_indices(engine_train_set_n + 1:engine_test_set_n + engine_train_set_n);

engine_train_set = engine_input(engine_train_indices, :);
engine_train_set = normalize(engine_train_set);
engine_train_set = engine_train_set(:,~any(ismissing(engine_train_set),1));
%engine_train_set = normalize_data(table2array(engine_train_set));
engine_train_set_cat = engine_label_categ(engine_train_indices);

engine_test_set = engine_input(engine_test_indices, :);
engine_test_set = normalize(engine_test_set);
engine_test_set = engine_test_set(:,~any(ismissing(engine_test_set),1));
%engine_test_set = normalize_data(table2array(engine_test_set));
engine_test_set_cat = engine_label_categ(engine_test_indices);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Apple Build initial test, cv vs % curves to determine test set sizing
% Use default hyper parameters for each model.
[app_tc_tree, app_cvc_tree] = learning_curves_tree(apple_train_set, apple_train_set_cat, "decision_tree", {}, true);
[app_tc_nn, app_cvc_nn] = learning_curves(apple_train_set, apple_train_set_cat, "neural_net", {});
[app_tc_svm, app_cvc_svm] = learning_curves(apple_train_set, apple_train_set_cat, "svm", {});
[app_tc_knn, app_cvc_knn] = learning_curves(apple_train_set, apple_train_set_cat, "knn", {'NumNeighbors', 2});
[app_tc_b_tree, app_cvc_b_tree] = learning_curves_b(apple_train_set, apple_train_set_cat, "boosted_tree", {"MaxNumSplits", 3}, {'Method', 'AdaBoostM1'});

%% Apple Plot initial train,c cv vs % curves
f1 = figure();
pc = 10:10:100;
plot(pc, app_tc_tree, '-k', pc, app_cvc_tree, '--k', 'LineWidth',1.5);
hold on; box on; grid on;
plot(pc, app_tc_nn, '-r', pc, app_cvc_nn, '--r', 'LineWidth',1.5);
plot(pc, app_tc_b_tree, '-b', pc, app_cvc_b_tree, '--b', 'LineWidth',1.5);
plot(pc, app_tc_svm, '-g', pc, app_cvc_svm, '--g', 'LineWidth',1.5);
plot(pc, app_tc_knn, '-m', pc, app_cvc_knn, '--m', 'LineWidth',1.5);
xlabel('% Train Set','FontSize',14)
ylabel('Misclassification Rate','FontSize',14)
title('Apple Quality Initial Learning Curves','FontSize',14)
ylim([0 inf])
legend('DT Train','DT CV', 'NN Train','NN CV', 'BDT Train','BDT CV', 'SVM','SVM CV', 'KNN Train', 'KNN CV', 'Location', 'bestoutside');
hold off

saveas(f1, 'init_apple_traincv.png')

%% Engine Build initial train, cv vs % curves to determine test set sizing
% Use default hyper parameters for each model.
[eng_tc_tree, eng_cvc_tree] = learning_curves_tree(engine_train_set, engine_train_set_cat, "decision_tree", {}, true);
[eng_tc_nn, eng_cvc_nn] = learning_curves(engine_train_set, engine_train_set_cat, "neural_net", {});
[eng_tc_svm, eng_cvc_svm] = learning_curves(engine_train_set, engine_train_set_cat, "svm", {});
[eng_tc_knn, eng_cvc_knn] = learning_curves(engine_train_set, engine_train_set_cat, "knn", {'NumNeighbors', 2});
[eng_tc_b_tree, eng_cvc_b_tree] = learning_curves_b(engine_train_set, engine_train_set_cat, "boosted_tree", {"MaxNumSplits", 3}, {'Method', 'AdaBoostM1'});

%% Engine Plot initial test,c cv vs % curves
f2 = figure();
pc = 10:10:100;
plot(pc, eng_tc_tree, '-k', pc, eng_cvc_tree, '--k', 'LineWidth',1.5);
hold on; box on; grid on;
plot(pc, eng_tc_nn, '-r', pc, eng_cvc_nn, '--r', 'LineWidth',1.5);
plot(pc, eng_tc_b_tree, '-b', pc, eng_cvc_b_tree, '--b', 'LineWidth',1.5);
plot(pc, eng_tc_svm, '-g', pc, eng_cvc_svm, '--g', 'LineWidth',1.5);
plot(pc, eng_tc_knn, '-m', pc, eng_cvc_knn, '--m', 'LineWidth',1.5);
xlabel('% Train Set','FontSize',14)
ylabel('Misclassification Rate','FontSize',14)
title('Aircraft Engine Servicing Initial Learning Curves','FontSize',14)
ylim([0 inf])
legend('DT Train','DT CV', 'NN Train','NN CV', 'BDT Train','BDT CV', 'SVM','SVM CV', 'KNN Train', 'KNN CV', 'Location', 'bestoutside');
hold off

saveas(f2, 'init_engine_traincv.png')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decision Tree Validation Curves
[apple_dt_val, apple_minleafsize, apple_prune_state] = validation_curve_tree(apple_train_set, apple_train_set_cat, 'Apple Quality Decision Tree Validation');
[engine_dt_val, engine_minleafsize, engine_prune_state] = validation_curve_tree(engine_train_set, engine_train_set_cat, 'Aircraft Engine Decision Tree Validation');

saveas(apple_dt_val, 'dt_val_apple.png')
saveas(engine_dt_val, 'dt_val_engine.png')
%% Boosted Decision Tree Validation Curves
[apple_b_val, apple_numtrees, apple_numsplits] = validation_curve_b(apple_train_set, apple_train_set_cat, 'Apple Quality Boosted DT Validation', apple_minleafsize);
[engine_b_val, engine_numtrees, engine_numsplits] = validation_curve_b(engine_train_set, engine_train_set_cat, 'Aircraft Engine Boosted DT Validation', engine_minleafsize);

saveas(apple_b_val, 'b_val_apple.png')
saveas(engine_b_val, 'b_val_engine.png')

%% KNN Validation Curves
[apple_knn_val, apple_K, apple_knn_distfunc] = validation_curve_knn(apple_train_set, apple_train_set_cat, 'Apple Quality KNN Validation');
[engine_knn_val, engine_K, engine_knn_distfunc] = validation_curve_knn(engine_train_set, engine_train_set_cat, 'Aircraft Engine KNN Validation');

saveas(apple_knn_val, 'knn_val_apple.png')
saveas(engine_knn_val, 'knn_val_engine.png')
%% SVM Validation Curves
rng(123456)
[apple_svm_val, apple_box, apple_svm_kernel] = validation_curve_svm(apple_train_set, apple_train_set_cat, 'Apple Quality SVM Validation');
[engine_svm_val, engine_box, engine_svm_kernel] = validation_curve_svm(engine_train_set, engine_train_set_cat, 'Aircraft Engine SVM Validation');

saveas(apple_svm_val, 'svm_val_apple.png')
saveas(engine_svm_val, 'svm_val_engine.png')
%% Neural Net Validation Curves
rng(123456)
[apple_nn_width_val, apple_layer_width] = validation_curve_nn_width(apple_train_set, apple_train_set_cat, 'Apple Quality NN Layer Width Validation');
[apple_nn_val, apple_layers, apple_activationfunc] = validation_curve_nn(apple_train_set, apple_train_set_cat, 'Apple Quality NN Validation', apple_layer_width);
[engine_nn_width_val, engine_layer_width] = validation_curve_nn_width(engine_train_set, engine_train_set_cat, 'Aircraft Engine NN Layer Width Validation');
[engine_nn_val, engine_layers, engine_activationfunc] = validation_curve_nn(engine_train_set, engine_train_set_cat, 'Aircraft Engine NN Validation', engine_layer_width);

saveas(apple_nn_width_val, 'nn_width_val_apple.png')
saveas(engine_nn_width_val, 'nn_width_val_engine.png')
saveas(apple_nn_val, 'nn_val_apple.png')
saveas(engine_nn_val, 'nn_val_engine.png')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compile Hyperparameters
% Decision Tree
apple_dt_hyp = {'MinLeafSize', apple_minleafsize, 'Prune', apple_prune_state};
engine_dt_hyp = {'MinLeafSize', engine_minleafsize, 'Prune', engine_prune_state};

% Boosted Decision Tree
apple_b_hyp1 = {'MinLeafSize', apple_minleafsize, 'Prune', apple_prune_state, 'MaxNumSplits', apple_numsplits};
apple_b_hyp2 = {'NumLearningCycles', apple_numtrees};
engine_b_hyp1 = {'MinLeafSize', engine_minleafsize, 'Prune', engine_prune_state, 'MaxNumSplits', engine_numsplits};
engine_b_hyp2 = {'NumLearningCycles', engine_numtrees};

% KNN
apple_knn_hyp = {'Distance', apple_knn_distfunc, 'NumNeighbors', apple_K};
engine_knn_hyp = {'Distance', engine_knn_distfunc, 'NumNeighbors', engine_K};

% SVM
apple_svm_hyp = {'KernelFunction', apple_svm_kernel, 'BoxConstraint', apple_box};
engine_svm_hyp = {'KernelFunction', engine_svm_kernel, 'BoxConstraint', engine_box};

% NN
apple_nn_hyp = {'Activations', apple_activationfunc, 'LayerSizes', apple_layers};
engine_nn_hyp = {'Activations', engine_activationfunc, 'LayerSizes', engine_layers};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run Train vs Validation using determined Hyperparameters
% Apple Quality
rng(123456)
[app_tc_tree, app_cvc_tree] = learning_curves_tree(apple_train_set, apple_train_set_cat, "decision_tree", apple_dt_hyp, strcmp(apple_prune_state, "On"));
[app_tc_nn, app_cvc_nn] = learning_curves(apple_train_set, apple_train_set_cat, "neural_net", apple_nn_hyp);
[app_tc_svm, app_cvc_svm] = learning_curves(apple_train_set, apple_train_set_cat, "svm", apple_svm_hyp);
[app_tc_knn, app_cvc_knn] = learning_curves(apple_train_set, apple_train_set_cat, "knn", apple_knn_hyp);
[app_tc_b_tree, app_cvc_b_tree] = learning_curves_b(apple_train_set, apple_train_set_cat, "boosted_tree", apple_b_hyp1, apple_b_hyp2);

%% Apple Plot train, cv vs % curves
final_apple_traincurves = figure();
pc = 10:10:100;
plot(pc, app_tc_tree, '-k', pc, app_cvc_tree, '--k', 'LineWidth',1.5);
hold on; box on; grid on;
plot(pc, app_tc_nn, '-r', pc, app_cvc_nn, '--r', 'LineWidth',1.5);
plot(pc, app_tc_b_tree, '-b', pc, app_cvc_b_tree, '--b', 'LineWidth',1.5);
plot(pc, app_tc_svm, '-g', pc, app_cvc_svm, '--g', 'LineWidth',1.5);
plot(pc, app_tc_knn, '-m', pc, app_cvc_knn, '--m', 'LineWidth',1.5);
xlabel('% Train Set','FontSize',14)
ylabel('Misclassification Rate','FontSize',14)
title('Apple Quality Final Learning Curves','FontSize',14)
ylim([0 inf])
legend('DT Train','DT CV', 'NN Train','NN CV', 'BDT Train','BDT CV', 'SVM','SVM CV', 'KNN Train', 'KNN CV', 'Location', 'bestoutside');
hold off

saveas(final_apple_traincurves, 'final_apple_traincv.png')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aircraft Engine Service train, cv vs % curves on final hyperparameters
% 
rng(123456)
[eng_tc_tree, eng_cvc_tree] = learning_curves_tree(engine_train_set, engine_train_set_cat, "decision_tree", engine_dt_hyp, strcmp(engine_prune_state, "On"));
[eng_tc_nn, eng_cvc_nn] = learning_curves(engine_train_set, engine_train_set_cat, "neural_net", engine_nn_hyp);
[eng_tc_svm, eng_cvc_svm] = learning_curves(engine_train_set, engine_train_set_cat, "svm", engine_svm_hyp);
[eng_tc_knn, eng_cvc_knn] = learning_curves(engine_train_set, engine_train_set_cat, "knn", engine_knn_hyp);
[eng_tc_b_tree, eng_cvc_b_tree] = learning_curves_b(engine_train_set, engine_train_set_cat, "boosted_tree", engine_b_hyp1, engine_b_hyp2);

%% Aircraft Engine Service Plot train,c cv vs % curves
final_engine_traincurves = figure();
pc = 10:10:100;
plot(pc, eng_tc_tree, '-k', pc, eng_cvc_tree, '--k', 'LineWidth',1.5);
hold on; box on; grid on;
plot(pc, eng_tc_nn, '-r', pc, eng_cvc_nn, '--r', 'LineWidth',1.5);
plot(pc, eng_tc_b_tree, '-b', pc, eng_cvc_b_tree, '--b', 'LineWidth',1.5);
plot(pc, eng_tc_svm, '-g', pc, eng_cvc_svm, '--g', 'LineWidth',1.5);
plot(pc, eng_tc_knn, '-m', pc, eng_cvc_knn, '--m', 'LineWidth',1.5);
xlabel('% Train Set','FontSize',14)
ylabel('Misclassification Rate','FontSize',14)
title('Aircraft Engine Servicing Final Learning Curves','FontSize',14)
ylim([0 inf])
legend('DT Train','DT CV', 'NN Train','NN CV', 'BDT Train','BDT CV', 'SVM','SVM CV', 'KNN Train', 'KNN CV', 'Location', 'bestoutside');
hold off

saveas(final_engine_traincurves, 'final_engine_traincv.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create Neural net iteration learning curves
% Apple
rng(123456);
apple_nn_iter_curve = nn_learning_curve(apple_test_set, apple_test_set_cat, apple_nn_hyp, 'Apple Quality NN Iteration Curves');
saveas(apple_nn_iter_curve, 'final_apple_nn_iter.png')

% Engine
engine_nn_iter_curve = nn_learning_curve(engine_test_set, engine_test_set_cat, engine_nn_hyp, 'Aircraft Engine Servicing NN Iteration Curves');
saveas(engine_nn_iter_curve, 'final_engine_nn_iter.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Time final models wall time to train and test
% Apple
rng(123456);
apple_dt_final_acc = @() test_accuracy(decision_tree(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_dt_hyp, strcmp(apple_prune_state, "On")), apple_test_set_cat);
apple_t_dt = timeit(apple_dt_final_acc);
apple_nn_final_acc = @() test_accuracy(neural_net(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_nn_hyp), apple_test_set_cat);
apple_t_nn = timeit(apple_nn_final_acc);
apple_svm_final_acc = @() test_accuracy(svm(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_svm_hyp), apple_test_set_cat);
apple_t_svm = timeit(apple_svm_final_acc);
apple_knn_final_acc = @() test_accuracy(knn(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_knn_hyp), apple_test_set_cat);
apple_t_knn = timeit(apple_knn_final_acc);
apple_b_final_acc = @() test_accuracy(boosted_tree(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_b_hyp1, apple_b_hyp2), apple_test_set_cat);
apple_t_b = timeit(apple_b_final_acc);

% engine
engine_dt_final_acc = @() test_accuracy(decision_tree(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_dt_hyp, strcmp(engine_prune_state, "On")), engine_test_set_cat);
engine_t_dt = timeit(engine_dt_final_acc);
engine_nn_final_acc = @() test_accuracy(neural_net(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_nn_hyp), engine_test_set_cat);
engine_t_nn = timeit(engine_nn_final_acc);
engine_svm_final_acc = @() test_accuracy(svm(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_svm_hyp), engine_test_set_cat);
engine_t_svm = timeit(engine_svm_final_acc);
engine_knn_final_acc = @() test_accuracy(knn(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_knn_hyp), engine_test_set_cat);
engine_t_knn = timeit(engine_knn_final_acc);
engine_b_final_acc = @() test_accuracy(boosted_tree(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_b_hyp1, engine_b_hyp2), engine_test_set_cat);
engine_t_b = timeit(engine_b_final_acc);

%% Generate and run final models on test sets
% Apple
rng(123456);
apple_dt_final_acc = test_accuracy(decision_tree(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_dt_hyp, strcmp(apple_prune_state, "On")), apple_test_set_cat);
apple_nn_final_acc = test_accuracy(neural_net(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_nn_hyp), apple_test_set_cat);
apple_svm_final_acc = test_accuracy(svm(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_svm_hyp), apple_test_set_cat);
apple_knn_final_acc = test_accuracy(knn(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_knn_hyp), apple_test_set_cat);
apple_b_final_acc = test_accuracy(boosted_tree(apple_train_set, apple_train_set_cat, ...
    apple_test_set, apple_b_hyp1, apple_b_hyp2), apple_test_set_cat);

% Engine
engine_dt_final_acc = test_accuracy(decision_tree(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_dt_hyp, strcmp(engine_prune_state, "On")), engine_test_set_cat);
engine_nn_final_acc = test_accuracy(neural_net(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_nn_hyp), engine_test_set_cat);
engine_svm_final_acc = test_accuracy(svm(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_svm_hyp), engine_test_set_cat);
engine_knn_final_acc = test_accuracy(knn(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_knn_hyp), engine_test_set_cat);
engine_b_final_acc = test_accuracy(boosted_tree(engine_train_set, engine_train_set_cat, ...
    engine_test_set, engine_b_hyp1, engine_b_hyp2), engine_test_set_cat);

%% Compile results in table
apple_sup_learning_results = [1-apple_dt_final_acc,1-apple_nn_final_acc,1-apple_b_final_acc,1-apple_svm_final_acc,1-apple_knn_final_acc];
engine_sup_learning_results = [1-engine_dt_final_acc,1-engine_nn_final_acc,1-engine_b_final_acc,1-engine_svm_final_acc,1-engine_knn_final_acc];
table_lbl_side = ["Apple Quality"; "Aircraft Engine Servicing"];
lbl_table = array2table(table_lbl_side, 'VariableNames', {'Dataset'});
results_table = array2table([apple_sup_learning_results; engine_sup_learning_results], 'VariableNames', {'Decision Tree', 'Neural Net','Boosted DT', 'SVM', 'KNN'});
combined_table = [lbl_table, results_table];
writetable(combined_table, 'supervised_learning_andy_cheng.xlsx')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Supervised Learning Methods
% Decision Tree
function tree_predict = decision_tree(X_train, Y_train, X_test, HYP, prune_c)
    tree_model = fitctree(X_train, Y_train, HYP{:});
    if prune_c
        tree_predict = predict(prune(tree_model), X_test);
    else
        tree_predict = predict(tree_model, X_test);
    end
end

% Neural Network
function net_predict = neural_net(X_train, Y_train, X_test, HYP)
    net_model = fitcnet(X_train, Y_train, HYP{:});
    net_predict = predict(net_model, X_test);
end

% Boosted Decision Tree
function b_tree_predict = boosted_tree(X_train, Y_train, X_test, HYP1, HYP2) %, num_cycles)
    t = templateTree(HYP1{:});
    b_tree_model = fitcensemble(X_train, Y_train, HYP2{:}, 'Learners', t);
    b_tree_predict = predict(b_tree_model, X_test);
end

% SVM
function svm_predict = svm(X_train, Y_train, X_test, HYP)
    svm_model = fitcsvm(X_train, Y_train, HYP{:});
    svm_predict = predict(svm_model, X_test);
end

% KNN
function knn_predict = knn(X_train, Y_train, X_test, HYP)
    knn_model = fitcknn(X_train, Y_train, HYP{:});
    knn_predict = predict(knn_model, X_test);
end

%% Helper functions
function accuracy = test_accuracy(prediction, truth)
    sum_correct = 0;
    if iscell(prediction)
        for i = 1:length(prediction)
            if prediction{i} == truth(i)
                sum_correct = sum_correct + 1;
            end
        end

    else
        for i = 1:length(prediction)
            if prediction(i) == truth(i)
                sum_correct = sum_correct + 1;
            end
        end
    end

    accuracy = sum_correct/length(truth);
end

function [data_norm] = normalize_data(data)
    data = data - min(data);
    norm = max(data);
    data_norm = data ./ norm;
    data_norm(isnan(data_norm)) = 0;
    data_norm = array2table(data_norm);
end

%% Analysis functions
% cross validation
function [cvMat] = cross_validation(X, Y, network_type, HYP)
    fh = str2func(network_type);
    order = unique(Y);
    func = @(Xtrain, ytrain, Xtest, ytest)confusionmat(ytest,...
        categorical(fh(Xtrain, ytrain, Xtest, HYP)), 'Order', order);
    partitions = cvpartition(Y, 'KFold', 5);
    confMat = crossval(func, table2array(X), Y, 'partition', partitions);
    cvMat = reshape(sum(confMat), length(order), length(order));
end

function [cvMat] = cross_validation_tree(X, Y, network_type, HYP, prune)
    fh = str2func(network_type);
    order = unique(Y);
    func = @(Xtrain, ytrain, Xtest, ytest)confusionmat(ytest,...
        categorical(fh(Xtrain, ytrain, Xtest, HYP, prune)), 'Order', order);
    partitions = cvpartition(Y, 'KFold', 5);
    confMat = crossval(func, table2array(X), Y, 'partition', partitions);
    cvMat = reshape(sum(confMat), length(order), length(order));
end

function [cvMat] = cross_validation_b(X, Y, network_type, HYP, HYP2)
    fh = str2func(network_type);
    order = unique(Y);
    func = @(Xtrain, ytrain, Xtest, ytest)confusionmat(ytest,...
        categorical(fh(Xtrain, ytrain, Xtest, HYP, HYP2)), 'Order', order);
    partitions = cvpartition(Y, 'KFold', 5);
    confMat = crossval(func, table2array(X), Y, 'partition', partitions);
    cvMat = reshape(sum(confMat), length(order), length(order));
end

% Get learning curves for cross validation and training accuracy
function [train_curve, cv_curve] = learning_curves(X, Y, network_type, HYP)
    train_curve = zeros(10,1);
    cv_curve = zeros(10,1);
    fh = str2func(network_type);
    for i = 1:10 % 10%-100%
        end_idx = floor(length(Y) * (i/10));
        % Train
        X_train = X(1:end_idx, :);
        Y_train = Y(1:end_idx);
        train_score = test_accuracy(fh(X_train, Y_train, X_train, HYP), Y_train);

        % CV
        cv_mat = cross_validation(X_train, Y_train, network_type, HYP);
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');
        
        % Add to curves
        train_curve(i) = 1 - train_score;
        cv_curve(i) = 1 - cv_score;

    end
end

% learning curve for tree w/ prune on/off
function [train_curve, cv_curve] = learning_curves_tree(X, Y, network_type, HYP, prune)
    train_curve = zeros(10,1);
    cv_curve = zeros(10,1);
    fh = str2func(network_type);
    for i = 1:10 % 10%-100%
        end_idx = floor(length(Y) * (i/10));
        % Train
        X_train = X(1:end_idx, :);
        Y_train = Y(1:end_idx);
        train_score = test_accuracy(fh(X_train, Y_train, X_train, HYP, prune), Y_train);

        % CV
        cv_mat = cross_validation_tree(X_train, Y_train, network_type, HYP, prune);
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');
        
        % Add to curves
        train_curve(i) = 1 - train_score;
        cv_curve(i) = 1 - cv_score;

    end
end

% learning curves vs % dataset for boosted tree
function [train_curve, cv_curve] = learning_curves_b(X, Y, network_type, HYP, HYP2)
    train_curve = zeros(10,1);
    cv_curve = zeros(10,1);
    fh = str2func(network_type);
    for i = 1:10 % 10%-100%
        end_idx = floor(length(Y) * (i/10));
        % Train
        X_train = X(1:end_idx, :);
        Y_train = Y(1:end_idx);
        train_score = test_accuracy(fh(X_train, Y_train, X_train, HYP, HYP2), Y_train);

        % CV
        cv_mat = cross_validation_b(X_train, Y_train, network_type, HYP, HYP2);
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');
        
        % Add to curves
        train_curve(i) = 1 - train_score;
        cv_curve(i) = 1 - cv_score;

    end
end

% decision tree
function [f1, min_range_idx, prune_state] = validation_curve_tree(X, Y, title_s)
    X_train = X;
    Y_train = Y;
    range = 1:100;
    train_curve = zeros(length(range),1);
    cv_curve = zeros(length(range),1);
    train_curve_p = zeros(length(range),1);
    cv_curve_p = zeros(length(range),1);
    for i = 1:length(range)
        train_score = test_accuracy(decision_tree(X_train, Y_train, X_train, {'MinLeafSize', range(i), 'Prune', 'Off'}, false), Y_train);
        train_score_p = test_accuracy(decision_tree(X_train, Y_train, X_train, {'MinLeafSize', range(i), 'Prune', 'On'}, true), Y_train);

        % CV
        cv_mat = cross_validation_tree(X_train, Y_train, 'decision_tree', {'MinLeafSize', range(i), 'Prune', 'Off'}, false);
        cv_mat_p = cross_validation_tree(X_train, Y_train, 'decision_tree', {'MinLeafSize', range(i), 'Prune', 'On'}, true);
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');
        cv_score_p = sum(diag(cv_mat_p)) / sum(cv_mat_p, 'all');

        % Add to curves
        train_curve(i) = 1 - train_score;
        train_curve_p(i) = 1 - train_score_p;
        cv_curve(i) = 1 - cv_score;
        cv_curve_p(i) = 1 - cv_score_p;
    end
    [min_prune, min_prune_range_idx] = min(cv_curve_p);
    [min_no_prune, min_no_prunerange_idx] = min(cv_curve);
    if min_prune < min_no_prune
        prune_state = "On";
        min_range_idx = min_prune_range_idx;
    else
        prune_state = "Off";
        min_range_idx = min_no_prunerange_idx;
    end
    f1 = figure();
    plot(range, train_curve, '.-k', range, cv_curve, '--k', 'LineWidth', 1.5);
    hold on; box on; grid on;
    plot(range, train_curve_p, 'xr')
    plot(range, cv_curve_p, '--r', 'LineWidth', 1.5);
    xlabel('Minimum Leaf Size', 'FontSize', 14)
    ylabel('Misclassification Rate','FontSize',14)
    title(char(title_s), 'FontSize', 14);
    legend('Train', 'CV', 'Pruned Train', 'Pruned CV', 'Location', 'best')
    hold off
end

% boosted tree
function [b1, num_trees, num_splits] = validation_curve_b(X, Y, title_s, min_leaf_size)
    maxNumSplits = [5, 10, 30, 50, 70];
    numMNS = numel(maxNumSplits);
    numTrees = 1:2:100;
    Mdl = cell(numMNS, 1);%,numLR);
    train_result = cell(numMNS, 1); %zeros(numLR, numMNS);
    kflc = cell(numMNS, 1);
    
    for j = 1:numMNS
        t = templateTree('MaxNumSplits',maxNumSplits(j), 'MinLeafSize', min_leaf_size, 'Prune', 'on');
        Mdl{j} = fitcensemble(X,Y,'NumLearningCycles',numTrees(end),...
            'Learners',t,'KFold',5);
        kflc{j} = kfoldLoss(Mdl{j}, 'Mode', 'cumulative');
        temp_curve = zeros(length(numTrees), 1);
        for i = 1:length(numTrees)
            Mdl_train = fitcensemble(X,Y,'NumLearningCycles',numTrees(i),...
                'Learners',t);
            temp_curve(i) = 1 -test_accuracy(predict(Mdl_train, X), Y);
        end
        train_result{j} = temp_curve;
    end
    min_splits = 1;
    min_splits_idx = 1;
    for k = 1:length(kflc)
        [min_temp, splits_idx] = min(kflc{k,:});
        if min_temp < min_splits
            min_splits = min_temp;
            num_splits = maxNumSplits(k);
            min_splits_idx = splits_idx;
        end
    end
    num_trees = min_splits_idx;
    lines_train = ["-k", "-r", "-g", "-b", "-m"];
    lines_cv = ["ok", "--r", "--g", "--b", "--m"];

    b1 = figure();
    plot(numTrees, train_result{1}, lines_train(1), 1:99, kflc{j}, lines_cv(1));
    box on; grid on; hold on;
    for i = 2:numMNS
        plot(numTrees, train_result{i}, lines_train(i), 1:99, kflc{i}, lines_cv(i));
    end
    xlabel('# of Weak Learners','FontSize',14)
    ylabel('Misclassification Rate','FontSize',14)
    title(char(title_s), 'FontSize', 14);
    legend('maxNumSplits=5, Train', 'maxNumSplits=5, CV', 'maxNumSplits=10, Train', ...
        'maxNumSplits=10, CV','maxNumSplits=30, Train', 'maxNumSplits=30, CV',...
        'maxNumSplits=50, Train', 'maxNumSplits=50, CV', ...
        'maxNumSplits=70, Train', 'maxNumSplits=70, CV', 'Location', 'best');
    hold off;
 
end

% KNN
function [b1, opt_k, opt_distfunc] = validation_curve_knn(X, Y, title_s)
    K = 1:2:100;
    dist_funcs = ["minkowski", "euclidean"];
    X_train = X;
    Y_train = Y;
    train_curve = zeros(length(K),1);
    cv_curve = zeros(length(K),1);
    train_curve_p = zeros(length(K),1);
    cv_curve_p = zeros(length(K),1);
    for i = 1:length(K)
        train_score = test_accuracy(knn(X_train, Y_train, X_train, {'Distance', dist_funcs(1), 'NumNeighbors', K(i)}), Y_train);
        train_score_p = test_accuracy(knn(X_train, Y_train, X_train, {'Distance', dist_funcs(2),'NumNeighbors', K(i)}), Y_train);

        % CV
        cv_mat = cross_validation(X_train, Y_train, 'knn', {'Distance', dist_funcs(1),'NumNeighbors', K(i)});
        cv_mat_p = cross_validation(X_train, Y_train, 'knn', {'Distance', dist_funcs(2),'NumNeighbors', K(i)});
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');
        cv_score_p = sum(diag(cv_mat_p)) / sum(cv_mat_p, 'all');

        % Add to curves
        train_curve(i) = 1 - train_score;
        train_curve_p(i) = 1 - train_score_p;
        cv_curve(i) = 1 - cv_score;
        cv_curve_p(i) = 1 - cv_score_p;
    end
    [min_euc, min_euc_range_idx] = min(cv_curve_p);
    [min_kow, min_kow_range_idx] = min(cv_curve);
    
    if min_euc < min_kow
        opt_k = K(min_euc_range_idx);
        opt_distfunc = dist_funcs(2);
    else
        opt_k = K(min_kow_range_idx);
        opt_distfunc = dist_funcs(1);
    end

    b1 = figure();
    plot(K, train_curve, '.-k', K, cv_curve, '--k', 'LineWidth', 1.5);
    hold on; box on; grid on;
    plot(K, train_curve_p, 'xr')
    plot(K, cv_curve_p, '--r', 'LineWidth', 1.5);
    xlabel('# of Nearest Neighbors', 'FontSize', 14)
    ylabel('Misclassification Rate','FontSize',14)
    title(char(title_s), 'FontSize', 14);
    legend('Minkowski Train', 'Minkowski CV', 'Euclidean Train', 'Euclidean CV', 'Location', 'best')
    hold off
end

% KNN
function [b1, opt_box, opt_kernel] = validation_curve_svm(X, Y, title_s)
    box_n = 1:10:500;
    kernels = ["linear", "rbf"];
    X_train = X;
    Y_train = Y;
    train_curve = zeros(length(box_n),1);
    cv_curve = zeros(length(box_n),1);
    train_curve_p = zeros(length(box_n),1);
    cv_curve_p = zeros(length(box_n),1);
    for i = 1:length(box_n)
        train_score = test_accuracy(svm(X_train, Y_train, X_train, {'KernelFunction', kernels(1), 'BoxConstraint', box_n(i)}), Y_train);
        train_score_p = test_accuracy(svm(X_train, Y_train, X_train, {'KernelFunction', kernels(2),'BoxConstraint', box_n(i)}), Y_train);

        % CV
        cv_mat = cross_validation(X_train, Y_train, 'svm', {'KernelFunction', kernels(1),'BoxConstraint', box_n(i)});
        cv_mat_p = cross_validation(X_train, Y_train, 'svm', {'KernelFunction', kernels(2),'BoxConstraint', box_n(i)});
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');
        cv_score_p = sum(diag(cv_mat_p)) / sum(cv_mat_p, 'all');

        % Add to curves
        train_curve(i) = 1 - train_score;
        train_curve_p(i) = 1 - train_score_p;
        cv_curve(i) = 1 - cv_score;
        cv_curve_p(i) = 1 - cv_score_p;
    end
    [min_rbf, min_rbf_range_idx] = min(cv_curve_p);
    [min_gaus, min_gaus_range_idx] = min(cv_curve);
    
    if min_rbf < min_gaus
        opt_box = box_n(min_rbf_range_idx);
        opt_kernel = kernels(2);
    else
        opt_box = box_n(min_gaus_range_idx);
        opt_kernel = kernels(1);
    end

    b1 = figure();
    plot(box_n, train_curve, '.-k', box_n, cv_curve, '--k', 'LineWidth', 1.5);
    hold on; box on; grid on;
    plot(box_n, train_curve_p, '-r')
    plot(box_n, cv_curve_p, '--r', 'LineWidth', 1.5);
    xlabel('Box Constraint', 'FontSize', 14)
    ylabel('Misclassification Rate','FontSize',14)
    title(char(title_s), 'FontSize', 14);
    legend('Linear Train', 'Linear CV', 'RBF Train', 'RBF CV', 'Location', 'best')
    hold off
end

function [b1, opt_width] = validation_curve_nn_width(X, Y, title_s)
    box_n = [5, 10:10:100];
    X_train = X;
    Y_train = Y;
    train_curve = zeros(length(box_n),1);
    cv_curve = zeros(length(box_n),1);
    for i = 1:length(box_n)
        train_score = test_accuracy(neural_net(X_train, Y_train, X_train, {'LayerSizes', [box_n(i)]}), Y_train);
        % CV
        cv_mat = cross_validation(X_train, Y_train, 'neural_net', {'LayerSizes', [box_n(i)]});
        cv_score = sum(diag(cv_mat)) / sum(cv_mat, 'all');

        % Add to curves
        train_curve(i) = 1 - train_score;
        cv_curve(i) = 1 - cv_score;
    end
    [~, min_loss_width_range_idx] = min(cv_curve);
    
    opt_width = box_n(min_loss_width_range_idx);

    b1 = figure();
    plot(box_n, train_curve, '.-k', box_n, cv_curve, '--k', 'LineWidth', 1.5);
    hold on; box on; grid on;
    xlabel('Fully Connected Layer Size', 'FontSize', 14)
    ylabel('Misclassification Rate','FontSize',14)
    title(char(title_s), 'FontSize', 14);
    legend('Train', 'CV', 'Location', 'best')
    hold off
end

function [b1, opt_depth, opt_activation] = validation_curve_nn(X, Y, title_s, opt_width)
    layers = {[opt_width], [opt_width, opt_width], [opt_width, opt_width, opt_width], [opt_width, opt_width, opt_width, opt_width],...
        [opt_width, opt_width, opt_width, opt_width, opt_width],[opt_width, opt_width, opt_width, opt_width, opt_width, opt_width],...
        [opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width], [opt_width, opt_width, opt_width, opt_width, opt_width,...
        opt_width, opt_width, opt_width], [opt_width,opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width],...
        [opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width, opt_width]};

    activations = ["relu", "tanh"];
    X_train = X;
    Y_train = Y;
    train_curve_relu = zeros(length(layers),1);
    cv_curve_relu = zeros(length(layers),1);
    train_curve_tanh = zeros(length(layers),1);
    cv_curve_tanh = zeros(length(layers),1);
    for i = 1:length(layers)
        train_score_relu = test_accuracy(neural_net(X_train, Y_train, X_train, {'Activations', activations(1), 'LayerSizes', layers{i}}), Y_train);
        train_score_tanh = test_accuracy(neural_net(X_train, Y_train, X_train, {'Activations', activations(2), 'LayerSizes', layers{i}}), Y_train);
        % CV
        cv_ma_relu = cross_validation(X_train, Y_train, 'neural_net', {'Activations', activations(1), 'LayerSizes', layers{i}});
        cv_score_relu = sum(diag(cv_ma_relu)) / sum(cv_ma_relu, 'all');
        cv_ma_tanh = cross_validation(X_train, Y_train, 'neural_net', {'Activations', activations(2), 'LayerSizes', layers{i}});
        cv_score_tanh = sum(diag(cv_ma_tanh)) / sum(cv_ma_tanh, 'all');

        % Add to curves
        train_curve_relu(i) = 1 - train_score_relu;
        cv_curve_relu(i) = 1 - cv_score_relu;
        train_curve_tanh(i) = 1 - train_score_tanh;
        cv_curve_tanh(i) = 1 - cv_score_tanh;
    end
    
    [min_relu, min_relu_range_idx] = min(cv_curve_relu);
    [min_tanh, min_tanh_range_idx] = min(cv_curve_tanh);
    
    if min_relu < min_tanh
        opt_depth = layers{min_relu_range_idx};
        opt_activation = activations(1);
    else
        opt_depth = layers{min_tanh_range_idx};
        opt_activation = activations(2);
    end

    b1 = figure();
    plot(1:10, train_curve_relu, '.-k', 1:10, cv_curve_relu, '--k', 'LineWidth', 1.5);
    hold on; box on; grid on;
    plot(1:10, train_curve_tanh, '.-r', 1:10, cv_curve_tanh, '--r', 'LineWidth', 1.5);
    xlabel('# of Fully Connected Layers', 'FontSize', 14)
    ylabel('Misclassification Rate','FontSize',14)
    title(char(title_s), 'FontSize', 14);
    legend('Relu Train', 'Relu CV', 'Tanh Train', 'Tanh CV', 'Location', 'best')
    hold off
end


function [b1] = nn_learning_curve(X, Y, HYP, title_s)
    c = cvpartition(Y,"Holdout",0.10);
    trainingIndices = training(c);
    validationIndices = test(c);
    X_train = X(trainingIndices,:);
    Y_train = array2table(Y(trainingIndices), 'VariableNames', {'tblValidation2'});
    tblValidation1 = X(validationIndices,:);
    tblValidation2 = Y(validationIndices,:);
    tblValidation = [tblValidation1, array2table(tblValidation2)];

    net_model = fitcnet(X_train, Y_train, HYP{:}, "ValidationData",tblValidation);
    iteration = net_model.TrainingHistory.Iteration;
    trainLosses = net_model.TrainingHistory.TrainingLoss;
    valLosses = net_model.TrainingHistory.ValidationLoss;
    
    b1 = figure();
    plot(iteration,trainLosses,'--k', iteration,valLosses, '--r', 'LineWidth', 1.5);
    hold on; grid on; box on;
    legend(["Training","Validation"], 'Location', 'best')
    xlabel("Iteration", 'FontSize',14)
    ylabel("Cross-Entropy Loss", 'FontSize',14)
    title(title_s, 'FontSize',14)
    hold off;
end
