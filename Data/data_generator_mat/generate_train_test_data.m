function [train_data_inputs_Npts,train_data_targets_Npts,test_data_inputs_Npts,test_data_targets_Npts] = generate_train_test_data(sigma,l_c,thld_KLE,number_of_Gauss_quadrature_points,train_data_size,test_data_size,h,V_basis,T_basis,BN_Dirichlet,beta,max_ite_GD,step_size_GD)
% generate training data (KLE coefficient, optimal pathwise control) and testing data (KLE coefficient)

format short e

%% 0. parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------------------------------%
% import functions
%---------------------------------------------------------------------------------------------------%
addpath('Models/Robust_Deterministic_Control/utils_mat');
addpath('Data/data_generator_mat')
addpath('Models/Pathwise_Control/models_mat')
%---------------------------------------------------------------------------------------------------%
% tunable parameters
%---------------------------------------------------------------------------------------------------%
rng(666,'twister'); rand_seed = rng; % for reproducibility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 1. generate (input) data -- realizations of log-normal coefficient via KLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n');
fprintf('1. solving KL eigenvalue problem ...');
%---------------------------------------------------------------------------------------------------%
% 1-1). truncated KL expansion of log-normal coefficient
%
% Discription:  (i)  coeff_sample_KLE_Npts(l,k) = the k-th realization of log-normal coefficient evaluated at the l-th grid point
%              (ii)  coeff_sample_KLE_Qpts(i,j,k) = the k-th realization of log-normal coefficient evaluated at the i-th Gauss quadrature point of the j-th element
%
% Remark 1: can use random variables uniformly distributed on [-2*sqrt(3),2*sqrt(3)] instead
% Remark 2: coeff_sample_KLE_Npts -> inputs of U-net; coeff_sample_KLE_Qpts -> inputs of discretized state and adjoint eqns
%---------------------------------------------------------------------------------------------------%
[coeff_sample_KLE_Npts] = generate_coefficient_realizations_KLE_normal(V_basis,T_basis,number_of_Gauss_quadrature_points,sigma,l_c,thld_KLE,rand_seed,train_data_size+test_data_size);
% generate training (input) and testing (input) data
train_data_inputs_Npts = coeff_sample_KLE_Npts(:,1:train_data_size);
test_data_inputs_Npts = coeff_sample_KLE_Npts(:,train_data_size+1:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 2. generate (target) data -- optimal pathwise control associated with coefficient realization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('2. solving distributed control problem with pathwise control ...');
%---------------------------------------------------------------------------------------------------%
% default setting
%---------------------------------------------------------------------------------------------------%
% homogeneous Dirichlet boundary condtion for governing system
%---------------------------------------------------------------------------------------------------%
% 2-1). initialize parallel computing environment
%---------------------------------------------------------------------------------------------------%
p = gcp('nocreate');
if isempty(p)
    fprintf('\n');
    parpool
else
    fprintf('\n');
    disp(['   parallel computing environment is initialized with CoreNum = ', num2str(p.NumWorkers)]);
end
%---------------------------------------------------------------------------------------------------%
% 2-2) solve distributed control problem with truncated KL expansion of log-normal coefficient
%---------------------------------------------------------------------------------------------------%
fprintf('2-1. preparing training dataset ...');
fprintf('\n');
start_path_ctrl_traindata = tic;
[~,~,train_data_targets_Npts,~] = path_ctrl_KLE_coeff_GD_solver(V_basis,T_basis,BN_Dirichlet,beta,train_data_inputs_Npts,step_size_GD,max_ite_GD,1);
end_path_ctrl_traindata = toc(start_path_ctrl_traindata);
fprintf('   elapsed time for finding path-ctrl on traindata = %d minutes and %f seconds', floor(end_path_ctrl_traindata/60), rem(end_path_ctrl_traindata,60));
fprintf('\n');

fprintf('2-2. preparing testing dataset ...'); % only for learning curves during testing process
fprintf('\n');
start_path_ctrl_testdata = tic;
[~,~,test_data_targets_Npts,~] = path_ctrl_KLE_coeff_GD_solver(V_basis,T_basis,BN_Dirichlet,beta,test_data_inputs_Npts,step_size_GD,max_ite_GD,0);
end_path_ctrl_testdata = toc(start_path_ctrl_testdata);
fprintf('   elapsed time for finding path-ctrl on testdata = %d minutes and %f seconds', floor(end_path_ctrl_testdata/60), rem(end_path_ctrl_testdata,60));
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3. save results for training, testing and plotting figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('3. save training and testing data (matrix form in mat format)');
% fprintf('\n');
% fprintf('   total number of training data = %d',train_data_size);
% fprintf('\n');
% fprintf('   total number of testing data = %d',test_data_size);
fprintf('\n');
%---------------------------------------------------------------------------------------------------%
save('Data/train_data_KLE_normal/Train_Data_KLE_Normal.mat','train_data_inputs_Npts','train_data_targets_Npts');
save('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat','test_data_inputs_Npts','test_data_targets_Npts');
%---------------------------------------------------------------------------------------------------%
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 4. convert data (matrix2tensor) for training and testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('4. save training and testing data (tensor form in mat format)');
fprintf('\n');
%---------------------------------------------------------------------------------------------------%
convert_data_matrix2tensor(h);
%---------------------------------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
