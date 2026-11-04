import numpy as np
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
from Data.data_generator_py.data_generator import generate_Gauss_point_local_triangle_py, triangular_local_basis_py

# ==============================================================================
# Tensor Conversions & Normalization
# ==============================================================================
def img_to_vec_torch(img_batch):
    flipped = torch.flip(img_batch, dims=[2])
    permuted = flipped.permute(0, 1, 3, 2)
    return permuted.reshape(img_batch.shape[0], -1).float()

def vec_to_img_torch(vec_batch, fem_data):
    B = vec_batch.shape[0]
    img_permuted = vec_batch.reshape(B, 1, fem_data['nx'], fem_data['ny'])
    img_flipped = img_permuted.permute(0, 1, 3, 2)
    return torch.flip(img_flipped, dims=[2])

def unnormalize_f_batch_torch(f_norm_vec_gpu, fem_data):
    return ((f_norm_vec_gpu * fem_data['max_f']) + fem_data['min_f'] + fem_data['mean_f_gpu'].squeeze()).float()

def normalize_f_batch_torch(f_raw_vec_gpu, fem_data):
    return ((f_raw_vec_gpu - fem_data['mean_f_gpu'].squeeze() - fem_data['min_f']) / fem_data['max_f'])

# ==============================================================================
# FEM Assembly & Precomputation
# ==============================================================================
def assemble_mass_matrix(fem_data):
    print("===> Assembling mass matrix M...")
    num_nodes = fem_data['num_nodes']
    T_basis = fem_data['T_basis']
    gauss_weights = fem_data['gauss_weights_precomputed']
    basis_vals = fem_data['basis_vals_precomputed']
    rows, cols, data = [], [], []

    for n in range(fem_data['num_elements']):
        element_nodes = T_basis[:3, n]
        b = basis_vals[n]
        w = gauss_weights[n]
        M_local = (b * w) @ b.T
        for i in range(3):
            for j in range(3):
                rows.append(element_nodes[i])
                cols.append(element_nodes[j])
                data.append(M_local[i, j])
                
    M_global = csc_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    print(f"===> Mass matrix M assembled! nnz={M_global.nnz}")
    return M_global

def get_U_target_nodes(fem_data):
    if 'U_target_nodes' not in fem_data:
        U_target_func = fem_data['U_target']
        V_basis = fem_data['V_basis']
        num_nodes = fem_data['num_nodes']
        
        node_coords = V_basis[:2, :]
        U_target_values = np.zeros(num_nodes)
        for i in range(num_nodes):
            U_target_values[i] = U_target_func(node_coords[0, i], node_coords[1, i])
        
        fem_data['U_target_nodes'] = U_target_values.reshape((fem_data['ny'], fem_data['nx']), order='F')
        fem_data['U_target_nodes_vec'] = U_target_values
    return fem_data['U_target_nodes']

def precompute_fem_constants(fem_data):
    print("===> Precomputing FEM constants...")
    num_elements = fem_data['num_elements']
    num_gauss_pts = fem_data['num_gauss_pts']
    element_vertices = fem_data['V_basis'][:2, fem_data['T_basis'][:3, :]]
    
    def compute_single_element(n):
        vertices = element_vertices[:, :, n]
        gauss_weights, gauss_points = generate_Gauss_point_local_triangle_py(num_gauss_pts, vertices.T)
        basis_vals = np.zeros((3, num_gauss_pts))
        for i in range(3):
            for j in range(num_gauss_pts):
                gp = gauss_points[j]
                basis_vals[i, j] = triangular_local_basis_py(gp[0], gp[1], vertices, 1, i, 0, 0)
        return gauss_weights, basis_vals

    results = Parallel(n_jobs=-1, batch_size=min(100, num_elements//10))(
        delayed(compute_single_element)(n) for n in range(num_elements)
    )
    fem_data['gauss_weights_precomputed'] = np.array([r[0] for r in results])
    fem_data['basis_vals_precomputed'] = np.array([r[1] for r in results])
    print("===> Precomputation complete!")
    return fem_data

# ==============================================================================
# PDE Solver & Objective Function
# ==============================================================================
def pde_solver(A_precomputed, u_numpy, v_numpy, fem_data):
    u_vector = u_numpy.flatten('F') if u_numpy.ndim > 1 else u_numpy
    v_vector = v_numpy.flatten('F') if v_numpy.ndim > 1 else v_numpy

    F = fem_data['mass_matrix'].dot(u_vector.astype(np.float64) - v_vector.astype(np.float64))
    F[fem_data['BN_Dirichlet']] = 0.0

    solution_vector = spsolve(A_precomputed, F, use_umfpack=True, permc_spec='MMD_AT_PLUS_A')
    return solution_vector.reshape((fem_data['ny'], fem_data['nx']), order='F')

def solve_for_one_sample(A_sample_bc, f_sample_raw, fem_data):
    f_sample_double = f_sample_raw.astype(np.float64)
    zero_vector = np.zeros_like(f_sample_double)
    u_sample = pde_solver(A_sample_bc, f_sample_double, zero_vector, fem_data)
    xi_sample = pde_solver(A_sample_bc, u_sample, fem_data['U_target_nodes'], fem_data)
    return u_sample, xi_sample

def compute_primal_objective(u_in, f_in, fem_data, beta):
    u_vec = u_in.flatten('F') if u_in.ndim > 1 else u_in
    f_vec = f_in.flatten('F') if f_in.ndim > 1 else f_in
    M = fem_data['mass_matrix']
    
    diff = u_vec - fem_data['U_target_nodes_vec']
    term_tracking = 0.5 * np.dot(diff, M.dot(diff))
    term_reg = 0.5 * beta * np.dot(f_vec, M.dot(f_vec))
    return term_tracking + term_reg

def weighted_mse_loss_raw(pred_raw, target_raw, M_sparse_torch):
    diff = pred_raw - target_raw
    Mdiff = torch.sparse.mm(M_sparse_torch, diff.t()).t()
    return 0.5 * torch.mean(torch.sum(diff * Mdiff, dim=1))