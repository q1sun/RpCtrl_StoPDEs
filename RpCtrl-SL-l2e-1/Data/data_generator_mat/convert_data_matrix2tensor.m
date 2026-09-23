function convert_data_matrix2tensor(h)

format short e

%% 1. data pre-processing: new_data = [ ( old_data - mean ) - min ] / ( max - min ) where min < 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('Data/train_data_KLE_normal/Train_Data_KLE_Normal.mat')
load('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat')
%-------------------------------------------------------------------------%
% 1-1. inputs data for training and testing
%-------------------------------------------------------------------------%
% compute mean 
mean_all_data_inputs_Npts = mean([train_data_inputs_Npts,test_data_inputs_Npts],2);
% input_data - mean
train_data_inputs_Npts = train_data_inputs_Npts - repmat(mean_all_data_inputs_Npts,1,size(train_data_inputs_Npts,2));
test_data_inputs_Npts = test_data_inputs_Npts - repmat(mean_all_data_inputs_Npts,1,size(test_data_inputs_Npts,2));
% compute min and max
min_all_data_inputs_Npts = min( min(min(train_data_inputs_Npts)), min(min(test_data_inputs_Npts)) );
max_all_data_inputs_Npts = max( max(max(train_data_inputs_Npts)), max(max(test_data_inputs_Npts)) ) - min( min(min(train_data_inputs_Npts)), min(min(test_data_inputs_Npts)) );
% [ ( input_data - mean ) - min ] / max
train_data_inputs_Npts = ( train_data_inputs_Npts - min_all_data_inputs_Npts * ones( size(train_data_inputs_Npts) ) ) / max_all_data_inputs_Npts;
test_data_inputs_Npts = ( test_data_inputs_Npts -min_all_data_inputs_Npts * ones( size(test_data_inputs_Npts) ) ) / max_all_data_inputs_Npts;
%-------------------------------------------------------------------------%
% 1-2-1. targets data for training and testing
%-------------------------------------------------------------------------%
% compute mean 
mean_all_data_targets_Npts = mean([train_data_targets_Npts,test_data_targets_Npts],2);
% input_data - mean
train_data_targets_Npts = train_data_targets_Npts - repmat(mean_all_data_targets_Npts,1,size(train_data_targets_Npts,2));
test_data_targets_Npts = test_data_targets_Npts - repmat(mean_all_data_targets_Npts,1,size(test_data_targets_Npts,2));
% compute min and max
min_all_data_targets_Npts = min( min(min(train_data_targets_Npts)), min(min(test_data_targets_Npts)) );
max_all_data_targets_Npts = max( max(max(train_data_targets_Npts)), max(max(test_data_targets_Npts)) ) - min( min(min(train_data_targets_Npts)), min(min(test_data_targets_Npts)) );
% [ ( input_data - mean ) - min ] / max
train_data_targets_Npts = ( train_data_targets_Npts - min_all_data_targets_Npts * ones( size(train_data_targets_Npts) ) ) / max_all_data_targets_Npts;
test_data_targets_Npts = ( test_data_targets_Npts -min_all_data_targets_Npts * ones( size(test_data_targets_Npts) ) ) / max_all_data_targets_Npts;
%-------------------------------------------------------------------------%
% 1-2-2. record for recovery
%-------------------------------------------------------------------------%
max_path_ctrls = max_all_data_targets_Npts;
min_path_ctrls = min_all_data_targets_Npts;
mean_path_ctrls = mean_all_data_targets_Npts;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 2. convert data: vector to tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------------------------------------%
% 2-1. training data -- vector to tensor
%-------------------------------------------------------------------------%
train_data_inputs_tensor = zeros(1/h(1)+1, 1/h(2)+1, size(train_data_inputs_Npts,2)); 
for i = 1 : size(train_data_inputs_Npts,2)
    train_data_inputs_tensor(:,:,i) = reshape(train_data_inputs_Npts(:,i), [], 1/h(1)+1);
    for j = 1 : 1/h(1)+1
        train_data_inputs_tensor(:,j,i) = fliplr(train_data_inputs_tensor(:,j,i)')';
    end
end

train_data_targets_tensor = zeros(1/h(1)+1, 1/h(2)+1, size(train_data_targets_Npts,2));
for i = 1 : size(train_data_targets_Npts,2)
    train_data_targets_tensor(:,:,i) = reshape(train_data_targets_Npts(:,i), [], 1/h(1)+1);
    for j = 1 : 1/h(1)+1
        train_data_targets_tensor(:,j,i) = fliplr(train_data_targets_tensor(:,j,i)')';
    end
end

% % inverse transform: tensor to vector without rescaling
% traindata_targets_tensor2vector = zeros(size(train_data_targets_Npts));
% traindata_targets_tensor_temp = zeros(1/h(1)+1, 1/h(2)+1, size(train_data_targets_Npts,2));
% for i = 1 : size(train_data_targets_Npts,2)
%     for j = 1 : 1/h(1)+1
%         traindata_targets_tensor_temp(:,j,i) = fliplr(traindata_targets_tensor(:,j,i)')';
%     end    
%     traindata_targets_tensor2vector(:,i) = reshape(traindata_targets_tensor_temp(:,:,i),(1/h(1)+1)^2,1);
% end

%-------------------------------------------------------------------------%
% 2-2. testing data -- vector to tensor
%-------------------------------------------------------------------------%
test_data_inputs_tensor = zeros(1/h(1)+1, 1/h(2)+1, size(test_data_inputs_Npts,2));
for i = 1 : size(test_data_inputs_Npts,2)
    test_data_inputs_tensor(:,:,i) = reshape(test_data_inputs_Npts(:,i), [], 1/h(1)+1);
    for j = 1 : 1/h(1)+1
        test_data_inputs_tensor(:,j,i) = fliplr(test_data_inputs_tensor(:,j,i)')';
    end
end

test_data_targets_tensor = zeros(1/h(1)+1, 1/h(2)+1, size(test_data_targets_Npts,2));
for i = 1 : size(test_data_targets_Npts,2)
    test_data_targets_tensor(:,:,i) = reshape(test_data_targets_Npts(:,i), [], 1/h(1)+1);
    for j = 1 : 1/h(1)+1
        test_data_targets_tensor(:,j,i) = fliplr(test_data_targets_tensor(:,j,i)')';
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 3. save results
%---------------------------------------------------------------------------------------------------%
save('Data/train_data_KLE_normal/Train_Data_KLE_Normal_Tensor.mat','train_data_inputs_tensor','train_data_targets_tensor','max_path_ctrls','min_path_ctrls','mean_path_ctrls');
save('Data/test_data_KLE_normal/Test_Data_KLE_Normal_Tensor.mat','test_data_inputs_tensor','test_data_targets_tensor','max_path_ctrls','min_path_ctrls','mean_path_ctrls');
%---------------------------------------------------------------------------------------------------%

end