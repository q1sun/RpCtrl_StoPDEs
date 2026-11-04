function plot_robpathctrl_samples()
% Generates a 3x4 sample comparison figure.
% Rows: Worst/Median/Best; Columns: Coefficient/Control/State/Error.

format short e

%% 1. Load Data
load('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat', 'test_data_inputs_Npts');
load('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobPathCtrl_KLE_Normal_Trained_Model.mat', 'rob_path_ctrl_OptCtrl_testdata');
rob_path_ctrl_OptCtrl = rob_path_ctrl_OptCtrl_testdata;

load('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobDetCtrl_vs_RobPathCtrl_KLE_Normal.mat', ...
     'rob_path_ctrl_OptState', 'rob_det_ctrl_TargetMatch_Index_testdata', 'V_basis');

%% 2. Sample Index Selection
use_auto_idx = false; 
if use_auto_idx
    num_samples = size(rob_path_ctrl_OptState, 2);
    idx_list = [rob_det_ctrl_TargetMatch_Index_testdata(1), ...
                rob_det_ctrl_TargetMatch_Index_testdata(round(num_samples/2)), ...
                rob_det_ctrl_TargetMatch_Index_testdata(num_samples)];
else
    idx_list = [886, 888, 900]; 
end

row_labels = {'Worst', 'Median', 'Best'};
col_titles = {'Coefficient \(a(x,\omega)\)', ...
              'Control \(f^*_{\omega,h}(x)\)', ...
              'State \(u^*_{\omega,h}(x)\)', ...
              'Error \(u^*_{\omega,h}(x)-U(x)\)'};

%% 3. Plotting Preparation
U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1)); 
Lx = max(V_basis(1,:)); Ly = max(V_basis(2,:));

% Interpolation grid
[qx, qy] = meshgrid(0:0.002:Lx, 0:0.002:Ly); 

nRows = 3; nCols = 4;
fig = figure('NumberTitle', 'off', 'Name', 'RobPathCtrl Sample Comparison');
set(fig, 'Color', 'w', 'Position', [100, 100, 1100, 700]);

% Compact layout parameters
left_margin = 0.05; right_margin = 0.04;   
bottom_margin = 0.06; top_margin = 0.06;
gap_h = 0.03; gap_v = 0.04;   

plot_width  = (1 - left_margin - right_margin - (nCols-1)*gap_h) / nCols;
plot_height = (1 - bottom_margin - top_margin - (nRows-1)*gap_v) / nRows;

%% 4. Plotting Loop
for r = 1:nRows
    idx = idx_list(r);
    for c = 1:nCols
        col = c - 1; row = r - 1;
        pos_left   = left_margin + col * (plot_width + gap_h);
        pos_bottom = bottom_margin + (nRows - 1 - row) * (plot_height + gap_v);
        
        subplot('Position', [pos_left, pos_bottom, plot_width, plot_height]);
        
        switch c
            case 1, data = test_data_inputs_Npts(:, idx);
            case 2, data = rob_path_ctrl_OptCtrl(:, idx);
            case 3, data = rob_path_ctrl_OptState(:, idx);
            case 4, data = rob_path_ctrl_OptState(:, idx) - U(V_basis(1,:), V_basis(2,:))';
        end
        
        F = scatteredInterpolant(V_basis(1,:)', V_basis(2,:)', data, 'linear', 'none');
        qz = F(qx, qy);
        
        imagesc(qz);
        colormap jet; axis image; axis off;
        
        % Set color axis limits
        if c == 3
            caxis([-2 2]); 
        elseif c == 4
            max_err = max(abs(data(:)));
            if max_err > 0, caxis([-max_err max_err]); end
        end
        
        % Add colorbar
        cb = colorbar('Location', 'eastoutside', 'FontSize', 8);
        cb.Position = [pos_left + plot_width + 0.005, pos_bottom, 0.012, plot_height];
        
        % Add titles and labels
        if r == 1
            title(col_titles{c}, 'Interpreter', 'latex', 'FontSize', 13);
        end
        if c == 1
            ylabel(row_labels{r}, 'Interpreter', 'latex', 'FontSize', 12, 'Rotation', 90);
        end
        if r == nRows && c <= 2
            xlabel('\(x\)', 'Interpreter', 'latex', 'FontSize', 12);
        end
    end
end

drawnow;
% saveas(gcf, 'robpathctrl_samples_compact.pdf');
end