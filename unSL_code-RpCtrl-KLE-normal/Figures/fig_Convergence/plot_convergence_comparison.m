function plot_convergence_comparison(sim_idx)
% Plot convergence curves: Pathwise, Robust Deterministic, and Physics-Informed U-Net.
% Input sim_idx: Best simulation index (1~6).

format short e

%% 1. Traditional Methods Data
steps_trad = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, ...
              1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, ...
              2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000];

j_pathwise = [3.718e-01, 8.632e-02, 2.845e-02, 1.402e-02, 9.160e-03, ...
              6.878e-03, 5.490e-03, 4.516e-03, 3.783e-03, 3.213e-03, ...
              2.761e-03, 2.396e-03, 2.100e-03, 1.857e-03, 1.655e-03, ...
              1.487e-03, 1.346e-03, 1.227e-03, 1.125e-03, 1.037e-03, ...
              9.620e-04, 8.964e-04, 8.390e-04, 7.885e-04, 7.437e-04, ...
              7.038e-04, 6.681e-04, 6.359e-04, 6.068e-04, 5.804e-04, 5.562e-04];

j_robust = [3.718e-01, 8.570e-02, 3.061e-02, 1.918e-02, 1.603e-02, ...
            1.463e-02, 1.373e-02, 1.307e-02, 1.256e-02, 1.216e-02, ...
            1.185e-02, 1.161e-02, 1.142e-02, 1.127e-02, 1.116e-02, ...
            1.106e-02, 1.099e-02, 1.093e-02, 1.089e-02, 1.085e-02, ...
            1.082e-02, 1.080e-02, 1.078e-02, 1.076e-02, 1.075e-02, ...
            1.074e-02, 1.073e-02, 1.072e-02, 1.072e-02, 1.071e-02, 1.071e-02];

%% 2. New Model Data (Read log from specified simulation)
log_path = sprintf('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_%d/log.txt', sim_idx);
if ~exist(log_path, 'file')
    error('Log file not found: %s', log_path);
end

data = readtable(log_path, 'HeaderLines', 1);

trainJ = data{:, 7};               
epochs_nn = (0:length(trainJ)-1)';  

% Initial random network state (Epoch 0)
random_nn_initial_J = data{1, 7};

%% 3. Plotting
figure('NumberTitle','off','Name','Convergence comparison');
hold on;

% Plot convergence curves
plot(steps_trad, j_pathwise, 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', 'Pathwise Gradient Descent');
plot(steps_trad, j_robust, 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'DisplayName', 'Robust Deterministic Control');
plot(epochs_nn, trainJ, 'Color', [0.85 0.325 0.098], 'LineWidth', 2.5, 'DisplayName', 'Physics-Informed U-Net');

% Scatter initial state markers
scatter(0, 3.718e-01, 60, 'k', 'x', 'LineWidth', 1.5, 'DisplayName', 'Initial State (\(f=0\))');
scatter(0, random_nn_initial_J, 60, 'r', 'x', 'LineWidth', 1.5, 'DisplayName', 'Initial State (Random NN)');

% Set axes properties
set(gca, 'YScale', 'log');
xlabel('Epoch Number', 'Interpreter', 'latex');
ylabel('Objective Function \(\mathcal{J}\)', 'Interpreter', 'latex');
xlim([0, max(steps_trad)]);
grid on; grid minor;
legend('Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 11);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', 14, 'TickLabelInterpreter', 'latex');
xticks(0:500:3000);

hold off;

%% 4. Save Figure
save_dir = 'Figures/fig_Convergence/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
saveas(gcf, fullfile(save_dir, 'convergence_comparison.png'));
fprintf('Figure saved to: %s\n', fullfile(save_dir, 'convergence_comparison.png'));

end