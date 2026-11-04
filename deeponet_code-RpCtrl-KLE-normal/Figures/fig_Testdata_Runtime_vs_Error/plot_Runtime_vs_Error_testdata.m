function plot_Runtime_vs_Error_testdata(plt_runtime_vs_error_testdata)

format short e

%% collect runtime and error for rob-det-ctrl, rob-path-ctrl and path-ctrl
time_offline = [287*60+15.70 402*60+29.86+46*60+33.64 0];
time_online = [0 0.51 76*60+34.66];

load('Figures/fig_Testdata_Runtime_vs_Error/Fig_Testdata_Target_Match_KLE_Normal.mat')

if plt_runtime_vs_error_testdata == 1
    
    %---------------------------------------------------------------------------------------------------%
    % (a) statistical results of target match and control cost
    %---------------------------------------------------------------------------------------------------%
    figure('NumberTitle','off','Name','rob_det_ctrl v.s. path_ctrl v.s. rob_path_ctrl using KLE coeff: target match, control cost, and runtime');
    
    subplot(1,2,1)
    h1 = histogram(rob_det_ctrl_TargetMatch);%,'FaceAlpha',0.5);
    hold on
    h2 = histogram(rob_path_ctrl_TargetMatch,20);%,'FaceAlpha',0.5);
    hold on
    h3 = histogram(path_ctrl_TargetMatch,20,'FaceAlpha',0.6);
    
    h1.FaceColor = 	[8.5000e-01   3.2500e-01   9.8000e-02];
    h2.FaceColor = [0   4.4700e-01   7.4100e-01];
    h3.FaceColor = [1 0.92 0.8];%[1 1 1]; % [0.9290   0.6940   0.1250]; %
    h1.Normalization = 'probability';
    h2.Normalization = 'probability';
    h3.Normalization = 'probability';
    set(gcf,'color','w')
    set(gca,'FontSize',18);
    % ax = axes;
    % ytickformat(ax, 'percentage');
    legend({'$\,\textnormal{RobDetCtrl}$','$\textnormal{RobPathCtrl}$','$\ \ \ \textnormal{PathCtrl}$'},'Interpreter','latex')
    ylabel({'$\textnormal{Percentage}$'},'Interpreter','latex')
    xlabel({'$\textnormal{Error of target match}$'},'Interpreter','latex')
    hold on
    
    
    subplot(1,2,2)
    h3 = histogram(rob_det_ctrl_CtrlCost);%,'FaceAlpha',0.5); %,'Orientation','horizontal');
    h3.BinWidth = 0.1;
    hold on
    h4 = histogram(rob_path_ctrl_CtrlCost);%,'FaceAlpha',0.5); %,'Orientation','horizontal');
    hold on
    h5 = histogram(path_ctrl_CtrlCost,'FaceAlpha',0.6);
    
    h3.FaceColor = 	[8.5000e-01   3.2500e-01   9.8000e-02];
    h4.FaceColor = [0   4.4700e-01   7.4100e-01];
    h5.FaceColor = [1 0.92 0.8];%[0.9290   0.6940   0.1250];%[1 1 1];
    h3.Normalization = 'probability';
    h4.Normalization = 'probability';
    h5.Normalization = 'probability';
    set(gcf,'color','w')
    set(gca,'FontSize',18);
    % ax = axes;
    % ytickformat(ax, 'percentage');
    legend({'$\,\textnormal{RobDetCtrl}$','$\textnormal{RobPathCtrl}$','$\ \ \ \textnormal{PathCtrl}$'},'Interpreter','latex')
    %legend({'$\ \| \hat{f}_{h,n}^*(x) \|_{L^2(D)}$','$\ \| f_{\omega,h}^*(x) \|_{L^2(D)}$'},'Interpreter','latex')
    ylabel({'\textnormal{Percentage}'},'Interpreter','latex')
    xlabel({'$\textnormal{Cost of desired control}$'},'Interpreter','latex')           
    %---------------------------------------------------------------------------------------------------%
    
end


fprintf('4. runtime:   rob_det_ctrl vs rob_path_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   (offline)    %6.3e       %6.3e      %6.3e',time_offline);
fprintf('\n');
fprintf('   (online)     %6.3e       %6.3e      %6.3e',time_online);
fprintf('\n');


% figure
% years = [2010 2012 2014]; % virtual x-coordinates
% 
% values1 = [[time_offline].' NaN(3,1)];
% values2 = [NaN(3,1) [time_online].'];
% years1 = repmat(years.', [1,2]);
%    
% bar(years1,values1, 'EdgeColor', 'b', 'LineWidth', 0.5);  
% 
% yyaxis right
% hB = bar(years1,values2, 'EdgeColor', 'r', 'LineWidth', 0.5);
% c = lines(2);
% for ind1 = 1:2
%   hB(ind1).FaceColor = c(ind1, :);
% end
% legend('Averaged matching error over test data', 'Runtime to achieve desired control online', 'Interpreter', 'latex')
% legend('Location','northwest')
% 
% set(gca,'xticklabel',{'RobDetCtrl','RobPathCtrl','PathCtrl'})
% hAxes.TickLabelInterpreter = 'latex'; 
% 
% ax = gca;
% ax.YAxis(1).Color = 'b';
% ax.YAxis(2).Color = 'r';
% 
% set(gcf,'color','w')
% set(gca,'FontSize',18); 


end