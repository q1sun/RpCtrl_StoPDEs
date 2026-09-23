function setup_KLE_normal


%% 0-1. problem description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------------------------------%
% 0-1-1. robust deterministic control
%---------------------------------------------------------------------------------------------------%                                       
% for almost everywhere w in \Omega, seek f(x) such that 
%           J_beta(f) = (1/2) E[ || u(x,w) - U(x) ||_{L^2(D)}^2 ] + (beta/2) || f(x) ||_{L^2(D)}^2 
% is minimized subject to 
%           - \nabla \cdot ( a(x,w) \nabla u(x,w) ) = f(x)    in D
%                                            u(x,w) = 0       on \partial D
%---------------------------------------------------------------------------------------------------%
% 0-1-2. pathwise control
%---------------------------------------------------------------------------------------------------%
% for each w in \Omega, seek f_w(x) such that 
%           J_beta(f_w) = (1/2) || u_w(x) - U(x) ||_{L^2(D)}^2  + (beta/2) || f_w(x) ||_{L^2(D)}^2 
% is minimized subject to 
%           - \nabla \cdot ( a_w(x) \nabla u_w(x) ) = f_w(x)    in D
%                                            u_w(x) = 0       on \partial D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 0-2. notation description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computation domain: D = [0,1] * [0,1];
% target function: U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1));
% log-normal coefficient: a(x,w) = exp( g(x,w) ) where E[g] = 0 and Cov_g(x,y) = sigma * exp( (-|x1-y1|-|x2-y2|) / l_c ) 
%---------------------------------------------------------------------------------------------------%  
% Input Parameters (problem setting): amplitude of Cov_g = sigma;
%                                     correlation length of Cov_g = l_c;
%---------------------------------------------------------------------------------------------------%  
% Input Parameters (numerical method): // linear element 
%                                      mesh size = h;                                     
%                                      quadrature rule for numerical integration = number_of_quadrature_points                             
%                                      // Monte Carlo method
%                                      sample size = MC_sample_size;                                     
%                                      // full-batch gradient descent 
%                                      maximum number of iterations = max_ite_GD; 
%                                      step size = step_size_GD;
%---------------------------------------------------------------------------------------------------%                                       
% Output Results: numerical robust deterministic control, numerical pathwise control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 0-3. hyperparameter configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('Models/Robust_Deterministic_Control/utils_mat');
%---------------------------------------------------------------------------------------------------%
% tunable parameters
%---------------------------------------------------------------------------------------------------%
format short e
h = [1/2^5,1/2^5]; % mesh size
sigma = 100; 
l_c = 0.2; 
train_data_size = 5000;
test_data_size = 1000;
beta = 10^(-10); % regularization constant for control cost
thld_KLE = 0.95; % energy threshold of truncated KLE
%---------------------------------------------------------------------------------------------------%
% default setting (linear element)
%---------------------------------------------------------------------------------------------------%
number_of_Gauss_quadrature_points = 7; 
plot_mesh = 0; % plot mesh, boundary nodes of linear element (activate = 1)
[V_basis,T_basis,BN_Dirichlet,~,~,~,~,~] = generate_mesh_linear_quadratic_element(h,plot_mesh);
%---------------------------------------------------------------------------------------------------%
% default setting (full batch gradient descent)
max_ite_GD = 3000; 
step_size_GD = 20;
%---------------------------------------------------------------------------------------------------%
fprintf('\n');
fprintf('\n');
fprintf('%%---------- hyperparameter configuration (start) ----------%%');
fprintf('\n');
fprintf('1. log-normal coeff:  amplitude  correlation  threshold');
fprintf('\n');
fprintf('                        %d      %6.1f       %6.2f',[sigma, l_c, thld_KLE]);
fprintf('\n');
fprintf('2. MC FE method:  mesh_size   quadrature   traindata_size   testdata_size');
fprintf('\n');
fprintf('                  %6.3e       %d            %d             %d',[h(1), number_of_Gauss_quadrature_points, train_data_size, test_data_size]);
fprintf('\n');
fprintf('3. full-batch GD method:  step_size   number_iteratios   beta');
fprintf('\n');
fprintf('                             %d             %d       %6.1e             %d',[step_size_GD, max_ite_GD, beta]);
fprintf('\n');
fprintf('%%---------- hyperparameter configuration (end) ----------%%');
fprintf('\n');
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 1. generate data for supervised learning with coefficient arising from truncated KLE 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('Data/data_generator_mat')
fprintf('\n');
fprintf('%%---------- data_generator using KLE_normal (start) ----------%%');
%---------------------------------------------------------------------------------------------------%
[train_data_inputs_Npts,train_data_targets_Npts,test_data_inputs_Npts,test_data_targets_Npts] = generate_train_test_data(sigma,l_c,thld_KLE,number_of_Gauss_quadrature_points,train_data_size,test_data_size,h,V_basis,T_basis,BN_Dirichlet,beta,max_ite_GD,step_size_GD);
%---------------------------------------------------------------------------------------------------%
fprintf('%%---------- data_generator using KLE_normal (end) ----------%%');
fprintf('\n');
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 2. robust deterministic control (baseline) using TKLE 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('Models/Robust_Deterministic_Control/models_mat')
fprintf('\n');
fprintf('%%---------- baseline: rob_det_ctrl using KLE_normal (start) ----------%%');
%---------------------------------------------------------------------------------------------------%
start_rob_det_ctrl_traindata = tic;
[~,~,rob_det_ctrl_OptCtrl,~] = rob_det_ctrl_KLE_coeff_GD_solver(V_basis,T_basis,BN_Dirichlet,beta,train_data_inputs_Npts,step_size_GD,max_ite_GD);
end_rob_det_ctrl_traindata = toc(start_rob_det_ctrl_traindata);
fprintf('   elapsed time for finding rob-det-ctrl on traindata = %d minutes and %f seconds', floor(end_rob_det_ctrl_traindata/60), rem(end_rob_det_ctrl_traindata,60));
fprintf('\n');
%---------------------------------------------------------------------------------------------------%
fprintf('2. save basline (robust deterministic control in matlab format)');
save('Checkpoints/RobDetCtrl_KLE_Normal.mat', 'rob_det_ctrl_OptCtrl');
%---------------------------------------------------------------------------------------------------%
fprintf('\n');
fprintf('%%---------- baseline: rob_det_ctrl using KLE_normal (end) ----------%%');
fprintf('\n');
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 3. results analysis for training data (1): rob_det_ctrl v.s. path_ctrl 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('Figures/fig_KLE')
addpath('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl')
fprintf('\n');
fprintf('%%---------- rob_det_ctrl vs path_ctrl on training data (start) ----------%%');
%---------------------------------------------------------------------------------------------------%
% tunable parameter
%---------------------------------------------------------------------------------------------------%
plt_KLE_eigenpair = 1;
plt_RobDetCtrl_vs_PathCtrl_traindata = 1;
%---------------------------------------------------------------------------------------------------%
% 3-1). plot eigenvalue and cumulative energy ratio for KL expansion of log-normal coefficient
%---------------------------------------------------------------------------------------------------%
plot_KLE_eigenpairs(plt_KLE_eigenpair);
%---------------------------------------------------------------------------------------------------%
% 3-2). target match & control cost: rob_det_ctrl v.s. path_ctrl
compare_RobDetCtrl_vs_PathCtrl_traindata(h);
plot_RobDetCtrl_vs_PathCtrl_traindata(plt_RobDetCtrl_vs_PathCtrl_traindata);
%---------------------------------------------------------------------------------------------------%
fprintf('%%---------- rob_det_ctrl vs path_ctrl on training data (end) ----------%%');
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



end
