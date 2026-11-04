function train_idx_sim = plot_learn_curves(plt_learn_curves)
% Plots learning curves for all simulations and finds the best model based on Test J.
% Note: Assumes the log file has 8 columns, where Train J is col 7 and Test J is col 8.

format short e

%% Read logs from all simulations
sim_path = 'Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_';
trainJ_all = [];
testJ_all = [];
epoch_counts = zeros(1, 6);

for i = 1:6
    fname = [sim_path num2str(i) '/log.txt'];
    if ~exist(fname, 'file')
        error('Log file not found: %s', fname);
    end
    data = readtable(fname, 'HeaderLines', 1);
    
    % IMPORTANT: If your new log file only has 3 columns (LR, Train J, Test J),
    % change 7 to 2 and 8 to 3 below.
    trainJ = data{:, 7}; 
    testJ  = data{:, 8}; 
    
    trainJ_all = [trainJ_all; trainJ'];
    testJ_all  = [testJ_all; testJ'];
    epoch_counts(i) = length(trainJ);
end

% Truncate to the minimum number of epochs across all simulations
minEpoch = min(epoch_counts);
if any(epoch_counts ~= minEpoch)
    fprintf('Warning: Different epoch counts. Truncating to %d epochs.\n', minEpoch);
    trainJ_all = trainJ_all(:, 1:minEpoch);
    testJ_all  = testJ_all(:, 1:minEpoch);
end

y_Unet_TrainLoss = trainJ_all;
y_Unet_TestLoss  = testJ_all;

%% Print Statistics
fprintf('\n%%---------- training results (start) ----------%%\n');

% Train J Stats
Unet_TrainLoss_mean = mean(min(y_Unet_TrainLoss, [], 2));
Unet_TrainLoss_best = min(min(y_Unet_TrainLoss, [], 2));
[train_idx_sim, idx_epoch] = find(y_Unet_TrainLoss == Unet_TrainLoss_best);

fprintf('1. Train J (mean of all simulations) = %6.3e\n', Unet_TrainLoss_mean);
fprintf('   Train J (best of all simulations) = %6.3e\n', Unet_TrainLoss_best);
fprintf('2. Model saved at the %d-th epoch of the %d-th simulation\n', idx_epoch(1), train_idx_sim(1));

% Test J Stats
Unet_TestLoss_mean = mean(min(y_Unet_TestLoss, [], 2));
Unet_TestLoss_best = min(min(y_Unet_TestLoss, [], 2));
[test_idx_sim, idx_epoch] = find(y_Unet_TestLoss == Unet_TestLoss_best);

fprintf('3. Test J (mean of all simulations) = %6.3e\n', Unet_TestLoss_mean);
fprintf('   Test J (best of all simulations) = %6.3e\n', Unet_TestLoss_best);
fprintf('   (The best Test J is achieved at the %d-th epoch of the %d-th simulation)\n', idx_epoch(1), test_idx_sim(1));

fprintf('%%---------- training results (end) ----------%%\n\n');

% Return the best simulation index based on Test J
train_idx_sim = test_idx_sim(1);

%% Plot Learning Curves
if plt_learn_curves == 1
    figure('NumberTitle', 'off', 'Name', 'Unet learning curves: training and testing');
    x = 1:size(y_Unet_TrainLoss, 2);
    
    A = shadedErrorBar(x, log10(y_Unet_TrainLoss), {@mean, @std}, 'lineprops', '-r');
    hold on
    B = shadedErrorBar(x, log10(y_Unet_TestLoss), {@mean, @std}, 'lineprops', '-b');
    hold on
    
    xlabel('Epoch', 'Interpreter', 'latex');
    ylabel('Objective \(\mathcal{J}\) on a \(\log_{10}\)-scale', 'Interpreter', 'latex');
    xlim([1, size(y_Unet_TrainLoss, 2)]);
    legend([A.mainLine, B.mainLine], {'Train', 'Test'}, 'Interpreter', 'latex');
    
    set(gcf, 'color', 'w');
    set(gca, 'FontSize', 16);
    set(gca, 'TickLabelInterpreter', 'latex');
    box on;
end

end