function [coeff_sample_KLE_Npts] = generate_coefficient_realizations_KLE_normal(V_basis,T_basis,number_of_Gauss_pts,sigma,l_c,thld_KLE,rand_seed,MC_sample_size)

format short e

%% problem setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  correlation length = l_c;
%         variance length = sigma;
%         mesh size = h;
% Solver:  piecewise linear element 
% Output:  eigenpairs, truncation number;
% Setting:  D = [0,1]*[0,1], Cov_g(x,y) = sigma * exp( (-|x1-y1|-|x2-y2|) / l_c ) ;
% reference: [2005 Schwab] FE for elliptic problems with sto coeff, Fig 6 (top left)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cov_g = @(x1,y1,x2,y2) sigma * exp( -(abs(x1-x2)+abs(y1-y2)) / l_c);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% linear element
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
basis_type = 1;
number_of_basis = 3;
number_of_elements = size(T_basis,2);    
number_of_nodes = size(V_basis,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% (a) solve generalized eigenvalue problem of KLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (a-1) assemble stiffness matrices 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lBaG = zeros(number_of_basis,number_of_Gauss_pts,number_of_elements);
H = zeros(number_of_basis,number_of_basis,number_of_elements);
K = zeros(number_of_nodes);
F = zeros(number_of_nodes);

% stiffness matrix
for n = 1 : number_of_elements
    %generate Gauss points for n-th element
    vertices_triangle = V_basis(1:2,T_basis(1:3,n));
    [Gauss_coefficient_local_triangle,Gauss_point_local_triangle] = generate_Gauss_point_local_triangle(number_of_Gauss_pts,vertices_triangle');
    
    for i = 1 : number_of_basis
        for j = 1 : number_of_Gauss_pts
            %lBaG(i,j,n) denotes the value corresponding to ith local base of nth element at local jth Gauss point  
            lBaG(i,j,n) = triangular_local_basis(Gauss_point_local_triangle(j,1),Gauss_point_local_triangle(j,2),vertices_triangle,basis_type,i,0,0);                     
        end
    end
        
    for k = 1 : number_of_basis
        H(k,:,n) = Gauss_coefficient_local_triangle * (repmat(lBaG(k,:,n),number_of_basis,1).*lBaG(:,:,n))';
    end
    K(T_basis(:,n),T_basis(:,n)) = H(:,:,n) + K(T_basis(:,n),T_basis(:,n));   
end

% covariance matrix
for i = 1 : number_of_nodes
    F(i,:) = Cov_g(V_basis(1,:),V_basis(2,:),V_basis(1,i),V_basis(2,i));
end
M = K*F*K;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (a-2) solve generalized eigenvalue problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[phi,D] = eig(M,K);
[lambda,index] = sort(diag(D),'descend');
phi = phi(:,index);

% normalize eigenfunctions
for i = 1 : number_of_nodes
    Normalize_matrices(1,i) = norm(phi(:,i));
end
Normalize_matrices = repmat(Normalize_matrices,number_of_nodes,1);
phi = phi./Normalize_matrices;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% (b) generate random variables ( can uniformly distributed on [-2*sqrt(3),2*sqrt(3)] instead )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (b-1) number of truncation terms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1 : size(lambda,1)
    energy_ratio(i) = sum(lambda(1:i)) / sum(lambda);
end
d = find(energy_ratio>thld_KLE);
dim_TKLE_log_a = d(1);

fprintf('\n');
fprintf('   total number of eigenpairs = %d',size(lambda,1));
fprintf('\n');
fprintf('   truncation number = %d',dim_TKLE_log_a);
fprintf('\n');
fprintf('   cumulative energy ration >=%6.2f',thld_KLE);
fprintf('\n');
fprintf('   normal distributed random variables are used where randseed = %d',rand_seed.Seed);
fprintf('\n');
fprintf('   (modify randseed in generate_train_test_data.m if necessary)');
fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (b-2) generate random variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(rand_seed); % for reproducibility
TKLE_rvs = randn(dim_TKLE_log_a,MC_sample_size); % normal distributed random variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% (c) generate sample realizations (truncated KLE) of log-normal coefficient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
truncated_lambda = lambda(1:dim_TKLE_log_a,:);
truncated_phi = phi(:,1:dim_TKLE_log_a);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c-1) coeff samples on grid points
log_a_sample_KLE_Npts = zeros(number_of_nodes,MC_sample_size);
for i = 1 : MC_sample_size
    log_a_sample_KLE_Npts(:,i) = sum( repmat(sqrt(truncated_lambda'),number_of_nodes,1) .* truncated_phi .* repmat(TKLE_rvs(:,i)',number_of_nodes,1),2);
end

coeff_sample_KLE_Npts = exp(log_a_sample_KLE_Npts);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % (c-2) coeff samples on quadrature points
% 
% % EaG(j,n,i) = i-th eigenfunction evaluated at j-th Gauss point corresoponding to the n-th element
% EaG = zeros(number_of_Gauss_pts,number_of_elements,dim_TKLE_log_a);
% 
% for i = 1 : dim_TKLE_log_a % index of eigenfunction    
%     for n =  1 : number_of_elements % index of element
%         for j =  1 : number_of_Gauss_pts % index of Gauss pt in n-th element
%             
%             EaG(j,n,i) = phi(T_basis(:,n),i)'*lBaG(:,j,n);
%             
%         end        
%     end
% end
% 
% % assemble random input data
% discrete_coeff = zeros(number_of_Gauss_pts,number_of_elements,MC_sample_size);
% for i = 1 : MC_sample_size
%     
%     % random variables multiply by the square root of eigenvalue
%     random_variable_temp = TKLE_rvs(:,i) .* sqrt(lambda(1:dim_TKLE_log_a));
%     
%     % compute discrete coeff value
%     for j = 1 : dim_TKLE_log_a       
%         discrete_coeff(:,:,i) = random_variable_temp(j) * EaG(:,:,j) + discrete_coeff(:,:,i);          
%     end        
% end
% 
% coeff_sample_KLE_Qpts = exp(discrete_coeff);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% (d) save results for plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('Figures/fig_KLE/Fig_Data_KLE.mat', 'dim_TKLE_log_a','lambda','energy_ratio','phi','V_basis');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end