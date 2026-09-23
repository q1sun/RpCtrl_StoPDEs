function plot_learn_curves(plt_learn_curves)

%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 = learning rate;
% 2 = training loss (TL);
% 3 = testing loss (VL);
%%%%%%%%%%%%%%%%%%%%%%%%%%

format short e
close all

%% load training results
Unet_R1 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_1/log.txt','', 1);
Unet_R2 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_2/log.txt','', 1);
Unet_R3 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_3/log.txt','', 1);
Unet_R4 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_4/log.txt','', 1);
Unet_R5 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_5/log.txt','', 1);
Unet_R6 = dlmread('Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_6/log.txt','', 1);
% training and testing loss
y_Unet_TrainLoss = [Unet_R1(:,2)';Unet_R2(:,2)';Unet_R3(:,2)';Unet_R4(:,2)';Unet_R5(:,2)';Unet_R6(:,2)'];
y_Unet_TrainLoss = y_Unet_TrainLoss/50;
y_Unet_TestLoss = [Unet_R1(:,3)';Unet_R2(:,3)';Unet_R3(:,3)';Unet_R4(:,3)';Unet_R5(:,3)';Unet_R6(:,3)'];
y_Unet_TestLoss = y_Unet_TestLoss/10;

%% statistical results
% train loss
Unet_TrainLoss_mean = mean(min(y_Unet_TrainLoss,[],2));
fprintf('   TrainLoss mean = %6.3e', Unet_TrainLoss_mean);
fprintf('\n')
Unet_TrainLoss_best = min(min(y_Unet_TrainLoss,[],2));
fprintf('   TrainLoss best = %6.3e', Unet_TrainLoss_best);
[idx_sim,idx_epoch] = find( y_Unet_TrainLoss == Unet_TrainLoss_best);
fprintf('\n')
fprintf('       achieved at simulation index = %d',idx_sim(1));
fprintf(' and epoch index = %d',idx_epoch(1));
fprintf('\n')
% test loss
Unet_TestLoss_mean = mean(min(y_Unet_TestLoss,[],2));
fprintf('   TestLoss mean = %6.3e', Unet_TestLoss_mean);
fprintf('\n')
Unet_TestLoss_best = min(min(y_Unet_TestLoss,[],2));
fprintf('   TestLoss best = %6.3e', Unet_TestLoss_best);
fprintf('\n')
[idx_sim,idx_epoch] = find( y_Unet_TestLoss == Unet_TestLoss_best);
fprintf('       achieved at simulation index = %d',idx_sim(1))
fprintf(' and epoch index = %d',idx_epoch(1));
fprintf('\n')

%% plot statistical results

if plt_learn_curves == 1
    figure('NumberTitle','off','Name','Unet learning curves: training and testing');
    
    %----------------------------------------------%
    % Unet Train Loss mean variance
    %----------------------------------------------%
    x = 1:size(y_Unet_TrainLoss,2);
    A = shadedErrorBar(x,log(y_Unet_TrainLoss),{@mean,@std},'lineprops', '-r');
    hold on
    %----------------------------------------------%
    % Unet Test Loss mean variance
    %----------------------------------------------%
    x = 1:size(y_Unet_TrainLoss,2);
    B = shadedErrorBar(x,log(y_Unet_TestLoss),{@mean,@std},'lineprops', '-b');
    hold on
    
    xlabel('Epoch','Interpreter','latex');
    ylabel('Loss on a log-scale','Interpreter','latex');
    xlim([1,size(y_Unet_TrainLoss,2)])
    legend([A.mainLine,B.mainLine],{'Train','Test'},'Interpreter','latex')
    set(gcf,'color','w')
    set(gca,'FontSize',16);
    set(gca,'TickLabelInterpreter','latex')
    box on
    %----------------------------------------------%
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end