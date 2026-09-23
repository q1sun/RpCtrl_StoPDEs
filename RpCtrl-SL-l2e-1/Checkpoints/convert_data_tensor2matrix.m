function [traindata_rob_path_ctrl_KLE_normal, testdata_rob_path_ctrl_KLE_normal] = convert_data_tensor2matrix(train_rob_path_ctrl, test_rob_path_ctrl)

format short e

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trained UNet output data -- tensor (NCHW) to vector (nN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------------------------------------------------- %
% 0. get the inverse transformation
% ----------------------------------------------------------------------- %
load('Data/train_data_KLE_normal/Train_Data_KLE_Normal_Tensor.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------- %
% 1. training data
% ----------------------------------------------------------------------- %
load('Data/train_data_KLE_normal/Train_Data_KLE_Normal.mat') % to get the full-batch dimension
load('Checkpoints/RobPathCtrl_KLE_Normal/simulation_1/Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat') % to get the mini-batch dimension
train_data_targets_Npts(:,size(trainbatch_rob_path_ctrl_OptCtrl_Tensor,1)+1:end) = [];
% reshape output tensor
tensor_temp = permute(train_rob_path_ctrl, [3,4,1,2]);
tensor_temp = squeeze(tensor_temp(:,:,:,1));

% tensor to vector
traindata_targets_tensor2vector = zeros(size(train_data_targets_Npts));
traindata_targets_tensor_temp = zeros(sqrt(size(train_data_targets_Npts,1)), sqrt(size(train_data_targets_Npts,1)), size(train_data_targets_Npts,2));
for i = 1 : size(train_data_targets_Npts,2)
    for j = 1 : sqrt(size(train_data_targets_Npts,1))
        traindata_targets_tensor_temp(:,j,i) = fliplr(tensor_temp(:,j,i)')';
    end
    traindata_targets_tensor2vector(:,i) = reshape(traindata_targets_tensor_temp(:,:,i),(sqrt(size(train_data_targets_Npts,1)))^2,1);
end

% save results
traindata_rob_path_ctrl_KLE_normal = traindata_targets_tensor2vector;
traindata_rob_path_ctrl_KLE_normal = traindata_rob_path_ctrl_KLE_normal * max_path_ctrls + min_path_ctrls * ones( size(train_data_targets_Npts) ) + repmat(mean_path_ctrls, 1, size(traindata_rob_path_ctrl_KLE_normal,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------- %
% 2. testing data
% ----------------------------------------------------------------------- %
load('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat') % to get the dimension
% ----------------------------------------------------------------------- %
% 1. testing data
% ----------------------------------------------------------------------- %
% reshape output tensor
tensor_temp = permute(test_rob_path_ctrl, [3,4,1,2]);
tensor_temp = squeeze(tensor_temp(:,:,:,1));

% tensor to vector
testdata_targets_tensor2vector = zeros(size(test_data_targets_Npts));
testdata_targets_tensor_temp = zeros(sqrt(size(test_data_targets_Npts,1)), sqrt(size(test_data_targets_Npts,1)), size(test_data_targets_Npts,2));
for i = 1 : size(test_data_targets_Npts,2)
    for j = 1 : sqrt(size(test_data_targets_Npts,1))
        testdata_targets_tensor_temp(:,j,i) = fliplr(tensor_temp(:,j,i)')';
    end
    testdata_targets_tensor2vector(:,i) = reshape(testdata_targets_tensor_temp(:,:,i),(sqrt(size(test_data_targets_Npts,1)))^2,1);
end

% save results
testdata_rob_path_ctrl_KLE_normal = testdata_targets_tensor2vector;
testdata_rob_path_ctrl_KLE_normal = testdata_rob_path_ctrl_KLE_normal * max_path_ctrls + min_path_ctrls * ones( size(test_data_targets_Npts) ) + repmat(mean_path_ctrls, 1, size(testdata_rob_path_ctrl_KLE_normal,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
