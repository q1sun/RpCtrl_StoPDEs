function plot_Runtime_vs_Error_testdata(plt_runtime_vs_error_testdata)

format short e

%% collect runtime and error for rob-det-ctrl, rob-path-ctrl and path-ctrl
time_offline = [287*60+15.70 402*60+29.86+28*60+39.10 0];
time_online = [0 0.62 76*60+34.66];

load('Figures/fig_Testdata_Runtime_vs_Error/Fig_Testdata_Target_Match_KLE_Normal.mat')

if plt_runtime_vs_error_testdata == 1
    
    %---------------------------------------------------------------------------------------------------%
    % (a) statistical results of target match and control cost
    %---------------------------------------------------------------------------------------------------%
    figure('NumberTitle','off', ...
       'Name','rob_det_ctrl v.s. path_ctrl v.s. rob_path_ctrl using KLE coeff: target match, control cost, and runtime', ...
       'Position', [100, 100, 1600, 400]);

    
    subplot(1,2,1)

    all_data1 = [rob_det_ctrl_TargetMatch(:); rob_path_ctrl_TargetMatch(:); path_ctrl_TargetMatch(:)];
    min_val1 = max(min(all_data1), 1e-10); 
    max_val1 = max(all_data1);

    num_bins = 32;
    edges1 = logspace(log10(min_val1), log10(max_val1), num_bins + 1);

    h1 = histogram(rob_det_ctrl_TargetMatch, edges1);
    hold on
    h2 = histogram(rob_path_ctrl_TargetMatch, edges1);
    h3 = histogram(path_ctrl_TargetMatch, edges1);

    h1.FaceColor = [0.8500   0.3250   0.0980]; 
    h3.FaceColor = [0        0.4470   0.7410]; 
    h2.FaceColor = [0.9290,  0.6940,  0.1250]; 
    
    h1.FaceAlpha = 0.6; 
    h2.FaceAlpha = 0.3; 
    h3.FaceAlpha = 0.3; 

    h1.Normalization = 'probability';
    h2.Normalization = 'probability';
    h3.Normalization = 'probability';

    set(gca, 'XScale', 'log'); 
    tick_vals = logspace(log10(min_val1), log10(max_val1), 6); 
    xticks(tick_vals);
    xticklabels(num2str(tick_vals', '%.2f')); 

    set(gcf,'color','w')
    set(gca,'FontSize',18);

    yt = get(gca, 'YTick');
    yt_labels = arrayfun(@(x) sprintf('%.0f%%', x*100), yt, 'UniformOutput', false);
    set(gca, 'YTickLabel', yt_labels);

    legend({'$\hat{f}_h^*(x)$','$f_{h,\omega}^{\,\theta^*}(x)$','$f_{h,\omega}^*(x)$'},'Interpreter','latex')
    ylabel({'$\textnormal{Percentage}$'},'Interpreter','latex')
    xlabel({'$\textnormal{Error of target match}$'},'Interpreter','latex')
    hold on



    subplot(1,2,2)
    
    h4 = histogram(rob_det_ctrl_CtrlCost);
    h4.BinWidth = 0.1;
    hold on
    h5 = histogram(rob_path_ctrl_CtrlCost);
    hold on
    h6 = histogram(path_ctrl_CtrlCost);

    h4.FaceColor = [0.8500   0.3250   0.0980]; 
    h6.FaceColor = [0        0.4470   0.7410]; 
    h5.FaceColor = [0.9290,  0.6940,  0.1250]; 
    
    h4.FaceAlpha = 0.6;
    h5.FaceAlpha = 0.3;
    h6.FaceAlpha = 0.3;

    h4.Normalization = 'probability';
    h5.Normalization = 'probability';
    h6.Normalization = 'probability';

    set(gcf,'color','w')
    set(gca,'FontSize',18);  
    yt = get(gca, 'YTick');
    yt_labels = arrayfun(@(x) sprintf('%.0f%%', x*100), yt, 'UniformOutput', false);
    set(gca, 'YTickLabel', yt_labels);

    legend({'$\hat{f}_h^*(x)$','$f_{h,\omega}^{\,\theta^*}(x)$','$f_{h,\omega}^*(x)$'},'Interpreter','latex')
    ylabel({'\textnormal{Percentage}'},'Interpreter','latex')
    xlabel({'$\textnormal{Cost of control function}$'},'Interpreter','latex')

    ax = gcf;
    exportgraphics(ax, 'Figures//fig_Testdata_Runtime_vs_Error//fig-TMatchHist-RobDet-RobPath-Path-KLE.pdf')
    
end


fprintf('4. runtime:   rob_det_ctrl vs rob_path_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   (offline)    %6.3e       %6.3e      %6.3e',time_offline);
fprintf('\n');
fprintf('   (online)     %6.3e       %6.3e      %6.3e',time_online);
fprintf('\n');


end