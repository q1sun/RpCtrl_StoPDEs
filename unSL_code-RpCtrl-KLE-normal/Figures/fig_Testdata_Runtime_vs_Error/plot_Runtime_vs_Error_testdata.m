function plot_Runtime_vs_Error_testdata(plt_runtime_vs_error_testdata)

format short e

%% Runtime data
time_offline = [287*60+15.70 2791.18 0];
time_online = [0 0.62 76*60+34.66];

load('Figures/fig_Testdata_Runtime_vs_Error/Fig_Testdata_Target_Match_KLE_Normal.mat')

if plt_runtime_vs_error_testdata == 1
    
    figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl v.s. rob_path_ctrl using KLE coeff: target match, control cost, and runtime');
    set(gcf,'color','w');
    
    % =====================================================================
    % (a) Target Match Histogram
    % =====================================================================
    subplot(1,2,1)
    
    % RobDetCtrl
    h1 = histogram(rob_det_ctrl_TargetMatch);
    h1.FaceColor = [0.49 0.18 0.56];    
    h1.FaceAlpha = 0.4;
    h1.EdgeColor = 'none';
    h1.Normalization = 'probability';
    hold on

    % RobPathCtrl
    h2 = histogram(rob_path_ctrl_TargetMatch, 20);
    h2.FaceColor = [0.00 0.45 0.74];
    h2.FaceAlpha = 0.5;
    h2.EdgeColor = 'none';
    h2.Normalization = 'probability';
    hold on

    % PathCtrl 
    h3 = histogram(path_ctrl_TargetMatch, 20);
    h3.FaceColor = [1.00 0.84 0.03];
    h3.FaceAlpha = 0.5;
    h3.EdgeColor = 'none';
    h3.Normalization = 'probability';
    
    set(gca,'FontSize',18);
    legend({'$\,\textnormal{RobDetCtrl}$','$\textnormal{RobPathCtrl}$','$\ \ \ \textnormal{PathCtrl}$'}, 'Interpreter','latex')
    ylabel('$\textnormal{Percentage}$', 'Interpreter','latex')
    xlabel('$\textnormal{Error of target match}$', 'Interpreter','latex')
    hold off
    
    % =====================================================================
    % (b) Control Cost Histogram
    % =====================================================================
    subplot(1,2,2)
    
    % RobDetCtrl 
    h3 = histogram(rob_det_ctrl_CtrlCost);
    h3.BinWidth = 0.2;
    h3.FaceColor = [0.49 0.18 0.56];
    h3.FaceAlpha = 0.4;
    h3.EdgeColor = 'none';
    h3.Normalization = 'probability';
    hold on

    % RobPathCtrl 
    h4 = histogram(rob_path_ctrl_CtrlCost);
    h4.FaceColor = [0.00 0.45 0.74];
    h4.FaceAlpha = 0.5;
    h4.EdgeColor = 'none';
    h4.Normalization = 'probability';
    hold on

    % PathCtrl
    h5 = histogram(path_ctrl_CtrlCost);
    h5.FaceColor = [1.00 0.84 0.03];
    h5.FaceAlpha = 0.5;
    h5.EdgeColor = 'none';
    h5.Normalization = 'probability';

    set(gca,'FontSize',18);
    legend({'$\,\textnormal{RobDetCtrl}$','$\textnormal{RobPathCtrl}$','$\ \ \ \textnormal{PathCtrl}$'}, 'Interpreter','latex')
    ylabel('\textnormal{Percentage}', 'Interpreter','latex')
    xlabel('$\textnormal{Cost of desired control}$', 'Interpreter','latex')           
    hold off
    
end

%% Print Runtime Statistics
fprintf('4. runtime:   rob_det_ctrl vs rob_path_ctrl vs path_ctrl\n');
fprintf('   (offline)    %6.3e       %6.3e      %6.3e\n', time_offline);
fprintf('   (online)     %6.3e       %6.3e      %6.3e\n', time_online);

end