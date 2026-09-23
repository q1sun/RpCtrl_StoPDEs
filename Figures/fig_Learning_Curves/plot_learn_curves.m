function train_idx_sim = plot_learn_curves(plt_learn_curves)

%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 = learning rate;
% 2 = training loss;
% 3 = testing loss;
%%%%%%%%%%%%%%%%%%%%%%%%%%

format short e

%% load training results
Unet_R1 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_1/log.txt','', 1);
Unet_R2 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_2/log.txt','', 1);
Unet_R3 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_3/log.txt','', 1);
Unet_R4 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_4/log.txt','', 1);
Unet_R5 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_5/log.txt','', 1);
Unet_R6 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_6/log.txt','', 1);
% training and testing loss
y_Unet_TrainLoss = [Unet_R1(:,2)';Unet_R2(:,2)';Unet_R3(:,2)';Unet_R4(:,2)';Unet_R5(:,2)';Unet_R6(:,2)'];
y_Unet_TestLoss = [Unet_R1(:,3)';Unet_R2(:,3)';Unet_R3(:,3)';Unet_R4(:,3)';Unet_R5(:,3)';Unet_R6(:,3)'];

% figure
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TrainLoss(1,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TrainLoss(2,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TrainLoss(3,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TrainLoss(4,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TrainLoss(5,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TrainLoss(6,:)))
% legend({'1','2','3','4','5','6'},'Interpreter','latex')

% figure
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TestLoss(1,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TestLoss(2,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TestLoss(3,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TestLoss(4,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TestLoss(5,:)))
% hold on
% plot(1:size(y_Unet_TrainLoss,2),log10(y_Unet_TestLoss(6,:)))
% legend({'1','2','3','4','5','6'},'Interpreter','latex')


%% statistical results
%---------------------------------------------------------------------------------------------------%
fprintf('\n');
fprintf('\n');
fprintf('%%---------- training results (start) ----------%%');
% train loss
fprintf('\n');
Unet_TrainLoss_mean = mean(min(y_Unet_TrainLoss,[],2));
fprintf('1. training loss (mean of all simulations) = %6.3e', Unet_TrainLoss_mean);
fprintf('\n')
Unet_TrainLoss_best = min(min(y_Unet_TrainLoss,[],2));
fprintf('   training loss (best of all simulations) = %6.3e', Unet_TrainLoss_best);
[train_idx_sim,idx_epoch] = find( y_Unet_TrainLoss == Unet_TrainLoss_best);
fprintf('\n')
fprintf('2. model saved at the %d', idx_epoch(1));
fprintf('-th epoch of the %d',train_idx_sim(1));
fprintf('-th simulation')
fprintf('\n')
% test loss
Unet_TestLoss_mean = mean(min(y_Unet_TestLoss,[],2));
fprintf('3. testing loss (mean of all simulations) = %6.3e', Unet_TestLoss_mean);
fprintf('\n')
Unet_TestLoss_best = min(min(y_Unet_TestLoss,[],2));
fprintf('   testing loss (best of all simulations) = %6.3e', Unet_TestLoss_best);
fprintf('\n')
[test_idx_sim,idx_epoch] = find( y_Unet_TestLoss == Unet_TestLoss_best);
fprintf('   (the best testing loss is achieved at the %d',idx_epoch(1))
fprintf('-th epoch of the %d',test_idx_sim(1));
fprintf('-th simulation)')
fprintf('\n')
fprintf('%%---------- training results (end) ----------%%');
fprintf('\n');
fprintf('\n');

train_idx_sim = train_idx_sim(1);

%% plot statistical results

if plt_learn_curves == 1
    figure('NumberTitle','off','Name','Unet learning curves: training and testing', 'Position', [100, 100, 1200, 550]);
    
    %----------------------------------------------%
    % Unet Train Loss mean variance
    %----------------------------------------------%
    x = 1:size(y_Unet_TrainLoss,2);
    A = shadedErrorBar(x,log10(y_Unet_TrainLoss),{@mean,@std},'lineprops', '-r');
    hold on
    %----------------------------------------------%
    % Unet Test Loss mean variance
    %----------------------------------------------%
    x = 1:size(y_Unet_TrainLoss,2);
    B = shadedErrorBar(x,log10(y_Unet_TestLoss),{@mean,@std},'lineprops', '-b');
    hold on
    
    xlabel('training epoch','Interpreter','latex');
    ylabel('loss on a $\textnormal{log}_{10}$-scale','Interpreter','latex');
    xlim([1,size(y_Unet_TrainLoss,2)])
    legend([A.mainLine,B.mainLine],{'Train Loss','Test Loss'},'Interpreter','latex')
    set(gcf,'color','w')
    set(gca,'FontSize',23);
    set(gca,'TickLabelInterpreter','latex')
    box on
    %----------------------------------------------%

    ax = gcf;
    exportgraphics(ax, 'Figures//fig_Learning_Curves//fig_learn_curves_SL.pdf')
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end