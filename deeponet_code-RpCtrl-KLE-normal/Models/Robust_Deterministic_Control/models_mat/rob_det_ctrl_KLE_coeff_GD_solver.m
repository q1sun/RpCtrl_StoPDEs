function [rob_det_ctrl_IteErr,rob_det_ctrl_IteObj,rob_det_ctrl_OptCtrl,rob_det_ctrl_OptState] = rob_det_ctrl_KLE_coeff_GD_solver(V_basis,T_basis,BN_Dirichlet,beta,train_data_inputs_Npts,step_size_GD,max_ite_GD)
% Discription: distributed elliptic control problem using robust deterministic control function and gradient descent method

%---------------------------------------------------------------------------------------------------%
fprintf('\n');
fprintf('1. solve distributed control problem with robust deterministic control');
%---------------------------------------------------------------------------------------------------%
% initialize parallel computing environment
%---------------------------------------------------------------------------------------------------%
p = gcp('nocreate');
if isempty(p)
    fprintf('\n');
    parpool
else
    fprintf('\n');
    disp(['   parallel computing environment is initialized with CoreNum = ', num2str(p.NumWorkers)]);
end

fprintf('1-1. training dataset');
fprintf('\n');
%---------------------------------------------------------------------------------------------------%

format short e

%% problem settings 
U = @(x,y) (sin(2*pi*x).*(cos(2*pi*y)-1)); % target function
MC_sample_size = size(train_data_inputs_Npts,2);

%% element type (linear element)
basis_type = 1;
number_of_basis = 3;
number_of_elements = size(T_basis,2);
number_of_nodes = size(V_basis,2);
number_of_Gauss_pts = 7;

%% assemble matrices (loop independent)
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

%% assemble matrices (loop dependent) & initialization
%%%%%%%%%%%%% initial force guess %%%%%%%%%%%%%
f_initial = zeros(size(V_basis,2),1); 
b_initial = M * f_initial;

% treat Dirichlet BC
for k = 1 : size(BN_Dirichlet,2)
    if BN_Dirichlet(1,k) == -1
       b_initial(BN_Dirichlet(2,k),1) = 0;
    end    
end

%%%%%%%%%%%%% stiffness matrices & initial state variate %%%%%%%%%%%%%
stiff_matrices = cell(1,MC_sample_size);
u_initial = zeros(number_of_nodes,MC_sample_size);
parfor p = 1 : MC_sample_size
      
    A1 = zeros(number_of_nodes);
    A2 = zeros(number_of_nodes);
    for r = 1 : number_of_nodes
        A1 = A1 + train_data_inputs_Npts(r,p) * B1(:,number_of_nodes*(r-1)+1:number_of_nodes*r);
        A2 = A2 + train_data_inputs_Npts(r,p) * B2(:,number_of_nodes*(r-1)+1:number_of_nodes*r);
    end   
    A = A1 + A2;
    
    % trear homogeneous Dirichlet boundary condition
    for q = 1 : size(BN_Dirichlet,2)
        if BN_Dirichlet(1,q) == -1
            A(BN_Dirichlet(2,q),:) = 0;
            A(BN_Dirichlet(2,q),BN_Dirichlet(2,q)) = 1;
        end
    end
    
    u_initial(:,p) = sparse(A) \ b_initial;
    stiff_matrices{1,p} = sparse(A);   
    
end

%%%%%%%%%%%%% initial objective %%%%%%%%%%%%%
M_cell = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    M_cell{1,k} = sparse(M);
end
diag_M_matrix = blkdiag(M_cell{1,:});

u_U_initial = u_initial - repmat(U(V_basis(1,:),V_basis(2,:))',1,MC_sample_size);
u_U_cell_initial = cell(1,MC_sample_size);
for k = 1 : MC_sample_size
    u_U_cell_initial{1,k} = sparse((u_U_initial(:,k))');
end
diag_u_U_vector_initial = blkdiag(u_U_cell_initial{1,:});

Obj_initial = 1/(2*MC_sample_size) * sum(diag(full(diag_u_U_vector_initial * diag_M_matrix * diag_u_U_vector_initial'))) + (beta/2)*(f_initial' * M * f_initial);

%% optimization using gradient descent
% fprintf('   total number of training data = %d',MC_sample_size);
% fprintf('\n');
% fprintf('   total number of iterations = %d',max_ite_GD);
% fprintf('\n');

index_ite = 1;
while index_ite < max_ite_GD + 1
       
    %%%%%%%%%%%%%% old information %%%%%%%%%%%%%%
    if index_ite == 1
        state_u_old = u_initial;
        f_old = f_initial;
        Obj_old = Obj_initial;
        fprintf('   iteration_index (live) = %d',index_ite);
        fprintf('     iteration_objective (live) = %6.3e',Obj_old);
        fprintf('\n');
    else
        state_u_old = state_u_new;
        f_old = f_new;
        Obj_old = Obj_new;
    end

    if mod(index_ite,100) == 0
        fprintf('   iteration_index (live) = %d',index_ite);
        fprintf('   iteration_objective (live) = %6.3e',Obj_old);
        fprintf('\n');
    end
    
    %%%%%%%%%%%%%% solve adjoint eqn %%%%%%%%%%%%%%
    % load vector
    u_U_old = state_u_old - repmat(U(V_basis(1,:),V_basis(2,:))',1,MC_sample_size);
    adj_RHS = M * u_U_old;
    
    % treat Dirichlet BC
    for k = 1 : size(BN_Dirichlet,2)
        if BN_Dirichlet(1,k) == -1
            adj_RHS(BN_Dirichlet(2,k),:) = 0;
        end
    end
    
    % solve adjoint solu
    parfor k = 1 : MC_sample_size
        adj_u(:,k) = sparse(stiff_matrices{1,k}) \ adj_RHS(:,k);
    end
    
    %%%%%%%%%%%%%% update control %%%%%%%%%%%%%%
    delta_f = - step_size_GD * ( beta * f_old + mean(adj_u,2) );
    f_new = f_old + delta_f;
    
    %%%%%%%%%%%%%% solve state eqn %%%%%%%%%%%%%%
    b_new = M * f_new;
    
    % treat Dirichlet BC
    for k = 1 : size(BN_Dirichlet,2)
        if BN_Dirichlet(1,k) == -1
            b_new(BN_Dirichlet(2,k),1) = 0;
        end
    end  
    
    % solve state solu
    parfor k = 1 : MC_sample_size
        state_u_new(:,k) = sparse(stiff_matrices{1,k}) \ b_new;
    end    
    
    %%%%%%%%%%%%%% evaluate obj_func %%%%%%%%%%%%%%
    u_U_new = state_u_new - repmat(U(V_basis(1,:),V_basis(2,:))',1,MC_sample_size);
    u_U_cell = cell(1,MC_sample_size);
    for k = 1 : MC_sample_size
        u_U_cell{1,k} = sparse((u_U_new(:,k))');
    end
    diag_u_U_vector = blkdiag(u_U_cell{1,:});

    Obj_new = 1/(2*MC_sample_size) * sum(diag(full(diag_u_U_vector * diag_M_matrix * diag_u_U_vector'))) + (beta/2)*(f_new' * M * f_new);

    %%%%%%%%%%%%%% calculate err %%%%%%%%%%%%%%
    relative_err = abs(Obj_new - Obj_old)/Obj_old;
    gradient_err = sqrt( (beta*f_old+mean(adj_u,2))'*M*(beta*f_old+mean(adj_u,2)) );
    
    %%%%%%%%%%%%%% save results %%%%%%%%%%%%%%
    rob_det_ctrl_IteErr(:,index_ite) = [index_ite;relative_err;gradient_err];
    rob_det_ctrl_IteObj(:,index_ite) = [index_ite;Obj_old];
    index_ite = index_ite + 1;
    
end

rob_det_ctrl_OptCtrl = f_new;
rob_det_ctrl_OptState = state_u_new;

save('Figures/fig_Traindata_RobDetCtrl_vs_PathCtrl/Fig_Traindata_RobDetCtrl_KLE_Normal.mat', 'rob_det_ctrl_OptCtrl','rob_det_ctrl_OptState','rob_det_ctrl_IteErr','rob_det_ctrl_IteObj','V_basis');

end