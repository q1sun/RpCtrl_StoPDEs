function test_KLE_normal

format short e

%% 1. ensemble stiffness matrices for testing data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------------------------------------------------------------------------------%
% 1-1) load coefficient realizations of testing data
%----------------------------------------------------------------------------------------------------------%
load('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat')
%----------------------------------------------------------------------------------------------------------%
% 1-2) assemble stiffness matrices
h = [1/2^5,1/2^5];
addpath('Models/Robust_Deterministic_Control/utils_mat');
%----------------------------------------------------------------------------------------------------------%
MC_sample_size = size(test_data_inputs_Npts,2);
[V_basis,T_basis,BN_Dirichlet,~,~,~,~,~] = generate_mesh_linear_quadratic_element(h,0);
basis_type = 1;
number_of_basis = 3;
number_of_elements = size(T_basis,2);
number_of_nodes = size(V_basis,2);
number_of_Gauss_pts = 7;

lBaG = zeros(number_of_basis,number_of_Gauss_pts,number_of_elements);
H = zeros(number_of_basis,number_of_basis,number_of_elements);
M = zeros(number_of_nodes);
lBaG1 = zeros(number_of_basis,number_of_Gauss_pts,number_of_elements);
lBaG2 = zeros(number_of_basis,number_of_Gauss_pts,number_of_elements);

% assemnble matrix for triple group due to finite element approximation of coefficient realizations
H1 = zeros(number_of_basis,number_of_basis,number_of_basis,number_of_elements);
% Remark: construction of B3 = zeros(number_of_node,number_of_node,number_of_node) is memory unaffordable, hence the spalloc structure is used instead
B1 = spalloc(number_of_nodes,number_of_nodes^2,number_of_nodes^2);

H2 = zeros(number_of_basis,number_of_basis,number_of_basis,number_of_elements);
% Remark: construction of B4 = zeros(number_of_node,number_of_node,number_of_node) is memory unaffordable, hence the spalloc structure is used instead
B2 = spalloc(number_of_nodes,number_of_nodes^2,number_of_nodes^2);
 
for n = 1 : number_of_elements    
    % generate local Gauss pts of n-th element
    vertices_triangle = V_basis(1:2,T_basis(1:3,n));
    [Gauss_coefficient_local_triangle,Gauss_point_local_triangle] = generate_Gauss_point_local_triangle(number_of_Gauss_pts,vertices_triangle');
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assemble matrix (int_D phi_i * phi_j dx)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : number_of_basis
        for j = 1 : number_of_Gauss_pts
            %lBaG(i,j,n) denotes the value corresponding to ith local base of nth element at local jth Gauss point
            lBaG(i,j,n) = triangular_local_basis(Gauss_point_local_triangle(j,1),Gauss_point_local_triangle(j,2),vertices_triangle,basis_type,i,0,0);
        end
    end
    
    for k = 1 : number_of_basis
        H(k,:,n) = Gauss_coefficient_local_triangle*(repmat(lBaG(k,:,n),number_of_basis,1).*lBaG(:,:,n))';
    end
    M(T_basis(:,n),T_basis(:,n)) = H(:,:,n) + M(T_basis(:,n),T_basis(:,n));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assemble matrix (int_D {partial phi_i w.r.t x} * {partial phi_j w.r.t. x} dx)
    for i = 1 : number_of_basis
        for j = 1 : number_of_Gauss_pts
            lBaG1(i,j,n) = triangular_local_basis(Gauss_point_local_triangle(j,1),Gauss_point_local_triangle(j,2),vertices_triangle,basis_type,i,1,0);%\partial(x)\phi_i \partial(x)\phi_j
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assemble matrix (int_D {partial phi_i w.r.t y} * {partial phi_j w.r.t. y} dy)
    for i = 1 : number_of_basis
        for j = 1 : number_of_Gauss_pts
            lBaG2(i,j,n) = triangular_local_basis(Gauss_point_local_triangle(j,1),Gauss_point_local_triangle(j,2),vertices_triangle,basis_type,i,0,1);%\partial(y)\phi_i \partial(y)\phi_j
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assemble B1 for matrix A1
    for r = 1 : number_of_basis
        for s = 1 : number_of_basis
           H1(s,:,r,n) = Gauss_coefficient_local_triangle*(repmat(lBaG(r,:,n),number_of_basis,1).*repmat(lBaG1(s,:,n),number_of_basis,1).*lBaG1(:,:,n))';   % \partial_x(\phi_i) \partial_x(\phi_j) \phi_p
        end
    end
    
    for r = 1 : number_of_basis
       B1(T_basis(:,n),T_basis(:,n) + number_of_nodes * (T_basis(r,n)-1)) = H1(:,:,r,n) + B1(T_basis(:,n),T_basis(:,n) + number_of_nodes * (T_basis(r,n)-1));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assemble B2 for matrix A2
    for r = 1 : number_of_basis
        for s = 1 : number_of_basis
           H2(s,:,r,n)= Gauss_coefficient_local_triangle*(repmat(lBaG(r,:,n),number_of_basis,1).*repmat(lBaG2(s,:,n),number_of_basis,1).*lBaG2(:,:,n))';   % \partial_y(\phi_i) \partial_y(\phi_j) \phi_p
        end
    end
    
    for r = 1 : number_of_basis
       B2(T_basis(:,n),T_basis(:,n) + number_of_nodes * (T_basis(r,n)-1)) = H2(:,:,r,n) + B2(T_basis(:,n),T_basis(:,n) + number_of_nodes * (T_basis(r,n)-1));   % \partial_y(\phi_i) \partial_y(\phi_j) \phi_p
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
end

test_data_inputs_Npts = test_data_inputs_Npts;

stiff_matrices = cell(1,MC_sample_size);
parfor p = 1 : MC_sample_size 
    
    A1 = zeros(number_of_nodes);
    A2 = zeros(number_of_nodes);
    for r = 1 : number_of_nodes
        A1 = A1 + test_data_inputs_Npts(r,p) * B1(:,number_of_nodes*(r-1)+1:number_of_nodes*r);
        A2 = A2 + test_data_inputs_Npts(r,p) * B2(:,number_of_nodes*(r-1)+1:number_of_nodes*r);
    end   
    A = A1 + A2;
    
    % trear homogeneous Dirichlet boundary condition
    for q = 1 : size(BN_Dirichlet,2)
        if BN_Dirichlet(1,q) == -1
            A(BN_Dirichlet(2,q),:) = 0;
            A(BN_Dirichlet(2,q),BN_Dirichlet(2,q)) = 1;
        end
    end
    stiff_matrices{1,p} = sparse(A);    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 2. evaluate target match using optimal rob_det_ctrl and rob_path_ctrl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('Checkpoints');
%----------------------------------------------------------------------------------------------------------%
% 2-1) results of the trained model
%----------------------------------------------------------------------------------------------------------%
% robust deterministic control
load('Checkpoints/RobDetCtrl_KLE_Normal.mat')
rob_det_ctrl = rob_det_ctrl_OptCtrl;
% robust pathwise control
load('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobPathCtrl_KLE_Normal_Trained_Model.mat')
rob_path_ctrl = rob_path_ctrl_OptCtrl_testdata; 
% pathwise control
load('Data/test_data_KLE_normal/Test_Data_KLE_Normal.mat')
path_ctrl = test_data_targets_Npts;

fprintf('\n');
fprintf('%%---------- Simulation Results of the Trained Model (start) ----------%%');

[rob_det_ctrl_TargetMatch, rob_det_ctrl_CtrlCost, rob_det_ctrl_OptState, ...
 rob_path_ctrl_TargetMatch, rob_path_ctrl_CtrlCost, rob_path_ctrl_OptState, ...
 path_ctrl_TargetMatch, path_ctrl_CtrlCost, path_ctrl_OptState, ...
 rob_det_ctrl_TargetMatch_Index_testdata] = evaluate_target_match_control_cost(rob_det_ctrl, rob_path_ctrl, path_ctrl, MC_sample_size, M, BN_Dirichlet, number_of_nodes, stiff_matrices, V_basis, test_data_inputs_Npts);

%----------------------------------------------------------------------------------------------------------%
% 2-2) save results: rob_det_ctrl vs rob_path_ctrl using KLE coeff
%----------------------------------------------------------------------------------------------------------%
save('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl/Fig_Testdata_RobDetCtrl_vs_RobPathCtrl_KLE_Normal.mat', 'rob_det_ctrl_TargetMatch','rob_det_ctrl_CtrlCost','rob_det_ctrl_TargetMatch_Index_testdata','rob_path_ctrl_TargetMatch','rob_path_ctrl_CtrlCost','rob_det_ctrl_OptState','rob_path_ctrl_OptState','V_basis');
save('Figures/fig_Testdata_Runtime_vs_Error/Fig_Testdata_Target_Match_KLE_Normal.mat', 'rob_det_ctrl_TargetMatch','rob_det_ctrl_CtrlCost','rob_path_ctrl_TargetMatch','rob_path_ctrl_CtrlCost','path_ctrl_TargetMatch','path_ctrl_CtrlCost','V_basis');
save('Figures/fig_Testdata_Runtime_vs_Error/Fig_Testdata_OptState_KLE_Normal.mat', 'rob_det_ctrl_OptState','rob_path_ctrl_OptState','path_ctrl_OptState');
%----------------------------------------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 3. plot results
addpath('Figures/fig_Testdata_RobDetCtrl_vs_RobPathCtrl');
plt_robdet_vs_robpath_testdata = 1;
plot_RobDetCtrl_vs_RobPathCtrl_testdata(plt_robdet_vs_robpath_testdata);
addpath('Figures/fig_Testdata_Runtime_vs_Error');
plt_runtime_vs_error_testdata = 1;
plot_Runtime_vs_Error_testdata(plt_runtime_vs_error_testdata);
fprintf('%%---------- Simulation Results of the Trained Model (end) ----------%%');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end



function [rob_det_ctrl_target_match, rob_det_ctrl_ctrl_cost, rob_det_ctrl_OptState, rob_path_ctrl_target_match, rob_path_ctrl_ctrl_cost, rob_path_ctrl_OptState, path_ctrl_target_match, path_ctrl_ctrl_cost, path_ctrl_OptState, index_target_match_rob_det_ctrl_KLE_coeff_testdata ] = evaluate_target_match_control_cost(rob_det_ctrl, rob_path_ctrl, path_ctrl, MC_sample_size, M, BN_Dirichlet, number_of_nodes, stiff_matrices, V_basis, test_data_inputs_Npts)

format short e

%% 1. solve elliptic BVPs with rob_det_ctrl, rob_path_ctrl and path_ctrl for testing data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1)); % target function
%----------------------------------------------------------------------------------------------------------%
% 2-1) solve elliptic BVP with rob_det_ctrl
%----------------------------------------------------------------------------------------------------------%
f_rob_det_ctrl = repmat(rob_det_ctrl,1,MC_sample_size); 
b_rob_det_ctrl = M * f_rob_det_ctrl;

for k = 1 : size(BN_Dirichlet,2)
    if BN_Dirichlet(1,k) == -1
       b_rob_det_ctrl(BN_Dirichlet(2,k),:) = 0;
    end    
end

u_rob_det_ctrl = zeros(number_of_nodes,MC_sample_size);
parfor p = 1 : MC_sample_size
        u_rob_det_ctrl(:,p) = stiff_matrices{1,p} \ b_rob_det_ctrl(:,p);
end

rob_det_ctrl_OptState = u_rob_det_ctrl;

M_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    M_cell{1,k} = sparse(M);
end
diag_M_matrix = blkdiag(M_cell{1,:});

% compute target match for each realization
u_U_rob_det_ctrl = u_rob_det_ctrl - repmat(U(V_basis(1,:),V_basis(2,:))',1,MC_sample_size);
u_U_rob_det_ctrl_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    u_U_rob_det_ctrl_cell{1,k} = sparse((u_U_rob_det_ctrl(:,k))');
end
diag_u_U_rob_det_ctrl_vector = blkdiag(u_U_rob_det_ctrl_cell{1,:});

rob_det_ctrl_target_match = diag(full(diag_u_U_rob_det_ctrl_vector * diag_M_matrix * diag_u_U_rob_det_ctrl_vector')).^(1/2);
[~,index_target_match_rob_det_ctrl_KLE_coeff_testdata] = sort(rob_det_ctrl_target_match,'descend');

% compute control cost for each realization
f_rob_det_ctrl_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    f_rob_det_ctrl_cell{1,k} = sparse((f_rob_det_ctrl(:,k))');
end
diag_f_rob_det_ctrl_vector = blkdiag(f_rob_det_ctrl_cell{1,:});

rob_det_ctrl_ctrl_cost = diag(full(diag_f_rob_det_ctrl_vector * diag_M_matrix * diag_f_rob_det_ctrl_vector')).^(1/2);
%---------------------------------------------------------------------------------------------------%
% 2-2) solve elliptic BVP with rob_path_ctrl
%----------------------------------------------------------------------------------------------------------%
b_rob_path_ctrl = M * rob_path_ctrl;

for k = 1 : size(BN_Dirichlet,2)
    if BN_Dirichlet(1,k) == -1
       b_rob_path_ctrl(BN_Dirichlet(2,k),:) = 0;
    end    
end

u_rob_path_ctrl = zeros(number_of_nodes,MC_sample_size);
parfor p = 1 : MC_sample_size
        u_rob_path_ctrl(:,p) = stiff_matrices{1,p} \ b_rob_path_ctrl(:,p);
end

rob_path_ctrl_OptState = u_rob_path_ctrl;

M_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    M_cell{1,k} = sparse(M);
end
diag_M_matrix = blkdiag(M_cell{1,:});

% compute target match for each realization
u_U_rob_path_ctrl = u_rob_path_ctrl - repmat(U(V_basis(1,:),V_basis(2,:))',1,MC_sample_size);
u_U_rob_path_ctrl_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    u_U_rob_path_ctrl_cell{1,k} = sparse((u_U_rob_path_ctrl(:,k))');
end
diag_u_U_rob_path_ctrl_vector = blkdiag(u_U_rob_path_ctrl_cell{1,:});

rob_path_ctrl_target_match = diag(full(diag_u_U_rob_path_ctrl_vector * diag_M_matrix * diag_u_U_rob_path_ctrl_vector')).^(1/2);
% [~,index_target_match_rob_path_ctrl_KLE_coeff_testdata] = sort(rob_path_ctrl_target_match,'descend');

% compute control cost for each realization
f_rob_path_ctrl_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    f_rob_path_ctrl_cell{1,k} = sparse((rob_path_ctrl(:,k))');
end
diag_f_rob_path_ctrl_vector = blkdiag(f_rob_path_ctrl_cell{1,:});

rob_path_ctrl_ctrl_cost = diag(full(diag_f_rob_path_ctrl_vector * diag_M_matrix * diag_f_rob_path_ctrl_vector')).^(1/2);
%---------------------------------------------------------------------------------------------------%
% 2-3) solve elliptic BVP with path_ctrl
%----------------------------------------------------------------------------------------------------------%
b_path_ctrl = M * path_ctrl;

for k = 1 : size(BN_Dirichlet,2)
    if BN_Dirichlet(1,k) == -1
       b_path_ctrl(BN_Dirichlet(2,k),:) = 0;
    end    
end

u_path_ctrl = zeros(number_of_nodes,MC_sample_size);
parfor p = 1 : MC_sample_size
        u_path_ctrl(:,p) = stiff_matrices{1,p} \ b_path_ctrl(:,p);
end

path_ctrl_OptState = u_path_ctrl;

M_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    M_cell{1,k} = sparse(M);
end
diag_M_matrix = blkdiag(M_cell{1,:});

% compute target match for each realization
u_U_path_ctrl = u_path_ctrl - repmat(U(V_basis(1,:),V_basis(2,:))',1,MC_sample_size);
u_U_path_ctrl_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    u_U_path_ctrl_cell{1,k} = sparse((u_U_path_ctrl(:,k))');
end
diag_u_U_path_ctrl_vector = blkdiag(u_U_path_ctrl_cell{1,:});

path_ctrl_target_match = diag(full(diag_u_U_path_ctrl_vector * diag_M_matrix * diag_u_U_path_ctrl_vector')).^(1/2);

% compute control cost for each realization
f_path_ctrl_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    f_path_ctrl_cell{1,k} = sparse((path_ctrl(:,k))');
end
diag_f_path_ctrl_vector = blkdiag(f_path_ctrl_cell{1,:});

path_ctrl_ctrl_cost = diag(full(diag_f_path_ctrl_vector * diag_M_matrix * diag_f_path_ctrl_vector')).^(1/2);
%---------------------------------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 2. print results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n');
fprintf('1. average target match and control cost: rob_det_ctrl vs rob_path_ctrl vs path_ctrl on testdata');
fprintf('\n');
fprintf('   target match (rob_det)  target match (rob_path)  target match (path)');
fprintf('\n');
fprintf('        %6.3e                %6.3e              %6.3e',[mean(rob_det_ctrl_target_match), mean(rob_path_ctrl_target_match), mean(path_ctrl_target_match)]);
fprintf('\n');
fprintf('   control cost (rob_det)  control cost (rob_path)  control cost (path)');
fprintf('\n');
fprintf('        %6.3e               %6.3e               %6.3e',[mean(rob_det_ctrl_ctrl_cost), mean(rob_path_ctrl_ctrl_cost), mean(path_ctrl_ctrl_cost)]);
fprintf('\n');
fprintf('2. target mismatch: rob_det_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   index of coeff sample         %d          %d          %d          %d',index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4));
fprintf('\n');
fprintf('   target match (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_target_match(index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4)));
fprintf('\n');
fprintf('   target match (rob_path)   %6.3e   %6.3e   %6.3e   %6.3e',rob_path_ctrl_target_match(index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4)));
fprintf('\n');
fprintf('   target match (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_target_match(index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4)));
fprintf('\n');
fprintf('   control cost (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_ctrl_cost(index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4)));
fprintf('\n');
fprintf('   control cost (rob_path)   %6.3e   %6.3e   %6.3e   %6.3e',rob_path_ctrl_ctrl_cost(index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4)));
fprintf('\n');
fprintf('   control cost (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_ctrl_cost(index_target_match_rob_det_ctrl_KLE_coeff_testdata(1:4)));
fprintf('\n');

num = size(test_data_inputs_Npts,2);
index = fliplr(fliplr(index_target_match_rob_det_ctrl_KLE_coeff_testdata(num-3:num))');
fprintf('3. target match: rob_det_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   index of coeff sample         %d          %d          %d          %d',index);
fprintf('\n');
fprintf('   target match (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_target_match(index));
fprintf('\n');
fprintf('   target match (rob_path)   %6.3e   %6.3e   %6.3e   %6.3e',rob_path_ctrl_target_match(index));
fprintf('\n');
fprintf('   target match (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_target_match(index));
fprintf('\n');
fprintf('   control cost (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_ctrl_cost(index));
fprintf('\n');
fprintf('   control cost (rob_path)   %6.3e   %6.3e   %6.3e   %6.3e',rob_path_ctrl_ctrl_cost(index));
fprintf('\n');
fprintf('   control cost (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_ctrl_cost(index));
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
