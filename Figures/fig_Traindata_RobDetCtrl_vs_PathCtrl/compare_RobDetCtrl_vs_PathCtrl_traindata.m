function compare_RobDetCtrl_vs_PathCtrl_traindata(h)

format short e

%% 1. rob_det_ctrl vs path_ctrl using KLE coeff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------------------------------%
% (1-1) evaluation of rob_det_ctrl using KLE coeff
%---------------------------------------------------------------------------------------------------%
% (1-1-1) assemble matrices for evaluation of target match
%---------------------------------------------------------------------------------------------------%
% mesh generation (linear element)
[V_basis,T_basis,~,~,~,~,~,~] = generate_mesh_linear_quadratic_element(h,0);
% element type (linear element)
basis_type = 1;
number_of_basis = 3;
number_of_elements = size(T_basis,2);
number_of_nodes = size(V_basis,2);
number_of_Gauss_pts = 7;
% assemble matrices (loop independent)
lBaG = zeros(number_of_basis,number_of_Gauss_pts,number_of_elements);
H = zeros(number_of_basis,number_of_basis,number_of_elements);
M = zeros(number_of_nodes);
 
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
end
%---------------------------------------------------------------------------------------------------%
% (1-1-2) assemble matrices for evaluation of target match
%---------------------------------------------------------------------------------------------------%
load('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Traindata_RobDetCtrl_KLE_Normal.mat');
U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1)); % target function
% 
% assembel matrices for each realization
M_cell = cell(1,size(rob_det_ctrl_OptState,2));
for k = 1 : size(rob_det_ctrl_OptState,2)
    M_cell{1,k} = sparse(M);
end
diag_M_matrix = blkdiag(M_cell{1,:});

% compute target match for each realization
u_U_rob_det = rob_det_ctrl_OptState - repmat(U(V_basis(1,:),V_basis(2,:))',1,size(rob_det_ctrl_OptState,2));
u_U_rob_det_cell = cell(1,size(rob_det_ctrl_OptState,2));

for k = 1 : size(rob_det_ctrl_OptState,2)
    u_U_rob_det_cell{1,k} = sparse((u_U_rob_det(:,k))');
end
diag_u_U_rob_det_vector = blkdiag(u_U_rob_det_cell{1,:});

rob_det_ctrl_TargetMatch = diag(full(diag_u_U_rob_det_vector * diag_M_matrix * diag_u_U_rob_det_vector')).^(1/2);

% compute control cost for each realization
f_rob_det_cell = cell(1,size(rob_det_ctrl_OptState,2));
for k = 1 : size(rob_det_ctrl_OptState,2)
    f_rob_det_cell{1,k} = sparse((rob_det_ctrl_OptCtrl)');
end
diag_f_rob_det_vector = blkdiag(f_rob_det_cell{1,:});

rob_det_ctrl_CtrlCost = diag(full(diag_f_rob_det_vector * diag_M_matrix * diag_f_rob_det_vector')).^(1/2);
%---------------------------------------------------------------------------------------------------%
% (1-1-3) index of target match in descending order
%---------------------------------------------------------------------------------------------------%
[~,rob_det_ctrl_TargetMatch_Index] = sort(rob_det_ctrl_TargetMatch,'descend');
%---------------------------------------------------------------------------------------------------%

%---------------------------------------------------------------------------------------------------%
% (1-2) evaluation of path_ctrl using KLE coeff
%---------------------------------------------------------------------------------------------------%
load('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Traindata_PathCtrl_KLE_Normal.mat');

% compute target match for each realization
u_U_path = path_ctrl_OptState - repmat(U(V_basis(1,:),V_basis(2,:))',1,size(path_ctrl_OptState,2));
u_U_path_cell = cell(1,size(path_ctrl_OptState,2));

for k = 1 : size(path_ctrl_OptState,2)
    u_U_path_cell{1,k} = sparse((u_U_path(:,k))');
end
diag_u_U_pathwise_vector = blkdiag(u_U_path_cell{1,:});

path_ctrl_TargetMatch = diag(full(diag_u_U_pathwise_vector * diag_M_matrix * diag_u_U_pathwise_vector')).^(1/2);

% compute control cost for each realization
f_path_cell = cell(1,size(path_ctrl_OptState,2));
for k = 1 : size(path_ctrl_OptState,2)
    f_path_cell{1,k} = sparse((path_ctrl_OptCtrl(:,k))');
end
diag_f_pathwise_vector = blkdiag(f_path_cell{1,:});

path_ctrl_CtrlCost = diag(full(diag_f_pathwise_vector * diag_M_matrix * diag_f_pathwise_vector')).^(1/2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2. save results: rob_det_ctrl vs path_ctrl using KLE coeff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Comparison_RobDetCtrl_vs_PathCtrl_KLE_Normal.mat','rob_det_ctrl_TargetMatch','rob_det_ctrl_CtrlCost','rob_det_ctrl_TargetMatch_Index','path_ctrl_TargetMatch','path_ctrl_CtrlCost');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3. print results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n');
fprintf('0. objective functional on training dataset: rob_det_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   obj_func (rob_det)  obj_func (path)');
fprintf('\n');
fprintf('       %6.3e        %6.3e',[rob_det_ctrl_IteObj(2,end), path_ctrl_IteObj(2,end)]);
fprintf('\n');
fprintf('1. average target match and control cost: rob_det_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   target match (rob_det)  target match (path)');
fprintf('\n');
fprintf('        %6.3e                %6.3e',[mean(rob_det_ctrl_TargetMatch), mean(path_ctrl_TargetMatch)]);
fprintf('\n');
fprintf('   control cost (rob_det)  control cost (path)');
fprintf('\n');
fprintf('        %6.3e               %6.3e',[mean(rob_det_ctrl_CtrlCost), mean(path_ctrl_CtrlCost)]);
fprintf('\n');
fprintf('2. target mismatch: rob_det_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   index of coeff sample       %d        %d        %d        %d',rob_det_ctrl_TargetMatch_Index(1:4));
fprintf('\n');
fprintf('   target match (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_TargetMatch(rob_det_ctrl_TargetMatch_Index(1:4)));
fprintf('\n');
fprintf('   target match (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_TargetMatch(rob_det_ctrl_TargetMatch_Index(1:4)));
fprintf('\n');
fprintf('   control cost (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_CtrlCost(rob_det_ctrl_TargetMatch_Index(1:4)));
fprintf('\n');
fprintf('   control cost (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_CtrlCost(rob_det_ctrl_TargetMatch_Index(1:4)));
fprintf('\n');

num = size(rob_det_ctrl_OptState,2);
index = fliplr(rob_det_ctrl_TargetMatch_Index(num-3:num));
fprintf('3. target match: rob_det_ctrl vs path_ctrl');
fprintf('\n');
fprintf('   index of coeff sample       %d        %d        %d        %d',index);
fprintf('\n');
fprintf('   target match (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_TargetMatch(index));
fprintf('\n');
fprintf('   target match (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_TargetMatch(index));
fprintf('\n');
fprintf('   control cost (rob_det)    %6.3e   %6.3e   %6.3e   %6.3e',rob_det_ctrl_CtrlCost(index));
fprintf('\n');
fprintf('   control cost (path)       %6.3e   %6.3e   %6.3e   %6.3e',path_ctrl_CtrlCost(index));
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end







