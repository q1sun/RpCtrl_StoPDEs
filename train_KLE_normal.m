function train_KLE_normal

format short e

%% 1. learning curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------------------------------------------------------------------------------%
% mean +- standard deviatioin over all simulations
%----------------------------------------------------------------------------------------------------------%
addpath('Figures/fig_Learning_Curves/')
plt_learn_curves = 1;
best_sim_idx = plot_learn_curves(plt_learn_curves);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 2. predictions of the trained model for training batch_data and testing data
%----------------------------------------------------------------------------------------------------------%
% 2-1) for the best model, conver format of network prediction from tensor to matrix
%----------------------------------------------------------------------------------------------------------%
addpath('Checkpoints/')
addpath('Models/Robust_Deterministic_Control/utils_mat');

if best_sim_idx == 1   
    
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_1/Testdata_RobPathCtrl_KLE_Normal_Tensor.mat') 
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_1/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat') 
    
elseif best_sim_idx == 2    
    
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_2/Testdata_RobPathCtrl_KLE_Normal_Tensor.mat') 
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_2/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat') 
    
elseif best_sim_idx == 3    
    
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_3/Testdata_RobPathCtrl_KLE_Normal_Tensor.mat') 
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_3/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat') 
    
elseif best_sim_idx == 4    
    
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_4/Testdata_RobPathCtrl_KLE_Normal_Tensor.mat') 
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_4/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat')  
    
elseif best_sim_idx == 5    
    
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_5/Testdata_RobPathCtrl_KLE_Normal_Tensor.mat') 
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_5/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat')
    
elseif best_sim_idx == 6    
    
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_6/Testdata_RobPathCtrl_KLE_Normal_Tensor.mat') 
    load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_6/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat') 
    
end
    
% convert data
[rob_path_ctrl_OptCtrl_trainbatch, rob_path_ctrl_OptCtrl_testdata] = convert_data_tensor2matrix(trainbatch_rob_path_ctrl_OptCtrl_Tensor, testdata_rob_path_ctrl_OptCtrl_Tensor);

% save results
save('Figures/fig_Traindata_RobPathCtrl_vs_PathCtrl/Fig_Traindata_RobPathCtrl_KLE_Normal_Trained_Model.mat','rob_path_ctrl_OptCtrl_trainbatch');
save('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobPathCtrl_KLE_Normal_Trained_Model.mat','rob_path_ctrl_OptCtrl_testdata');

%----------------------------------------------------------------------------------------------------------%
% 2-2) ground-truth vs prediction
%----------------------------------------------------------------------------------------------------------%
addpath('Figures/fig_Traindata_RobPathCtrl_vs_PathCtrl/')
plt_path_vs_robpath_traindata = 1;
plot_RobPathCtrl_vs_PathCtrl_traindata(plt_path_vs_robpath_traindata);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
