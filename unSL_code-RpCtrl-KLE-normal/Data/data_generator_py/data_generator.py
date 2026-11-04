import numpy as np
import scipy.linalg
import torch

# ==============================================================================
# Reference basis functions for linear triangular elements
# ==============================================================================
def triangular_reference_basis_py(x, y, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    r = 0.0
    if basis_type == 1:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1: r = 1 - x - y
            elif basis_index == 2: r = x
            elif basis_index == 3: r = y
        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1: r = -1.0
            elif basis_index == 2: r = 1.0
            elif basis_index == 3: r = 0.0
        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1: r = -1.0
            elif basis_index == 2: r = 0.0
            elif basis_index == 3: r = 1.0
    return r

def triangular_local_basis_py(x, y, vertices, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    J_11 = vertices[0, 1] - vertices[0, 0]; J_12 = vertices[0, 2] - vertices[0, 0]
    J_21 = vertices[1, 1] - vertices[1, 0]; J_22 = vertices[1, 2] - vertices[1, 0]
    J_det = J_11 * J_22 - J_12 * J_21
    if abs(J_det) < 1e-12: return 0.0
    
    x_hat = (J_22 * (x - vertices[0, 0]) - J_12 * (y - vertices[1, 0])) / J_det
    y_hat = (-J_21 * (x - vertices[0, 0]) + J_11 * (y - vertices[1, 0])) / J_det
    py_basis_index = basis_index + 1
    
    if derivative_degree_x == 0 and derivative_degree_y == 0:
        r = triangular_reference_basis_py(x_hat, y_hat, basis_type, py_basis_index, 0, 0)
    elif derivative_degree_x == 1 and derivative_degree_y == 0:
        term1 = triangular_reference_basis_py(x_hat, y_hat, basis_type, py_basis_index, 1, 0) * J_22
        term2 = triangular_reference_basis_py(x_hat, y_hat, basis_type, py_basis_index, 0, 1) * (-J_21)
        r = (term1 + term2) / J_det
    elif derivative_degree_x == 0 and derivative_degree_y == 1:
        term1 = triangular_reference_basis_py(x_hat, y_hat, basis_type, py_basis_index, 1, 0) * (-J_12)
        term2 = triangular_reference_basis_py(x_hat, y_hat, basis_type, py_basis_index, 0, 1) * J_11
        r = (term1 + term2) / J_det
    else: 
        r = 0.0
    return r

# ==============================================================================
# Gauss quadrature points for triangular elements
# ==============================================================================
def generate_Gauss_point_local_triangle_py(Gauss_point_number, vertices_triangle):
    area_det = np.linalg.det(np.vstack([np.ones(3), vertices_triangle.T]))
    
    if Gauss_point_number == 3:
        gauss_weights = np.array([1/6, 1/6, 1/6]) * area_det
        transform_matrix = np.array([[1-1/2-0, 1/2, 0], 
                                     [1-1/2-1/2, 1/2, 1/2], 
                                     [1-0-1/2, 0, 1/2]])
        gauss_points = transform_matrix @ vertices_triangle
        
    elif Gauss_point_number == 4:
        gauss_weights = np.array([(1-1/np.sqrt(3))/8, (1-1/np.sqrt(3))/8, (1+1/np.sqrt(3))/8, (1+1/np.sqrt(3))/8]) * area_det
        transform_matrix = np.array([
            [1-(1/np.sqrt(3)+1)/2-(1-1/np.sqrt(3))*(1+1/np.sqrt(3))/4, (1/np.sqrt(3)+1)/2, (1-1/np.sqrt(3))*(1+1/np.sqrt(3))/4],
            [1-(1/np.sqrt(3)+1)/2-(1-1/np.sqrt(3))*(1-1/np.sqrt(3))/4, (1/np.sqrt(3)+1)/2, (1-1/np.sqrt(3))*(1-1/np.sqrt(3))/4],
            [1-(-1/np.sqrt(3)+1)/2-(1+1/np.sqrt(3))*(1+1/np.sqrt(3))/4, (-1/np.sqrt(3)+1)/2, (1+1/np.sqrt(3))*(1+1/np.sqrt(3))/4],
            [1-(-1/np.sqrt(3)+1)/2-(1+1/np.sqrt(3))*(1-1/np.sqrt(3))/4, (-1/np.sqrt(3)+1)/2, (1+1/np.sqrt(3))*(1-1/np.sqrt(3))/4]
        ])
        gauss_points = transform_matrix @ vertices_triangle
        
    elif Gauss_point_number == 7:
        weights = np.array([0.225, 0.12593918054482717, 0.12593918054482717, 0.12593918054482717, 
                            0.13239415278850616, 0.13239415278850616, 0.13239415278850616])
        gauss_weights = weights / 2 * area_det
        transform_matrix = np.array([
            [1-1/3-1/3, 1/3, 1/3], 
            [1-0.79742698535308720-0.10128650732345633, 0.79742698535308720, 0.10128650732345633],
            [1-0.10128650732345633-0.79742698535308720, 0.10128650732345633, 0.79742698535308720],
            [1-0.10128650732345633-0.10128650732345633, 0.10128650732345633, 0.10128650732345633],
            [1-0.05971587178976981-0.47014206410511505, 0.05971587178976981, 0.47014206410511505],
            [1-0.47014206410511505-0.05971587178976981, 0.47014206410511505, 0.05971587178976981],
            [1-0.47014206410511505-0.47014206410511505, 0.47014206410511505, 0.47014206410511505]
        ])
        gauss_points = transform_matrix @ vertices_triangle
    else: 
        raise ValueError(f"Gauss point number {Gauss_point_number} not implemented.")
        
    return gauss_weights, gauss_points

# ==============================================================================
# Karhunen-Loève Expansion (KLE) generator for log-normal coefficients
# ==============================================================================
def generate_coefficient_realizations_kle_normal_py(V_basis, T_basis, number_of_Gauss_pts, sigma, l_c, thld_KLE, seed, MC_sample_size):
    if np.min(T_basis) == 1: 
        T_basis_py = T_basis - 1
    else: 
        T_basis_py = T_basis
        
    cov_g = lambda x1, y1, x2, y2: sigma * np.exp(-(np.abs(x1 - x2) + np.abs(y1 - y2)) / l_c)
    basis_type = 1
    number_of_basis = 3
    number_of_elements = T_basis_py.shape[1]
    number_of_nodes = V_basis.shape[1]
    
    # Assemble matrices K and M
    K = np.zeros((number_of_nodes, number_of_nodes))
    for n in range(number_of_elements):
        element_nodes = T_basis_py[:3, n]
        vertices_coords_2x3 = V_basis[:2, element_nodes]
        vertices_coords_3x2 = vertices_coords_2x3.T
        gauss_weights, gauss_points = generate_Gauss_point_local_triangle_py(number_of_Gauss_pts, vertices_coords_3x2)
        
        lBaG = np.zeros((number_of_basis, number_of_Gauss_pts))
        for i in range(number_of_basis):
            for j in range(number_of_Gauss_pts):
                lBaG[i, j] = triangular_local_basis_py(gauss_points[j, 0], gauss_points[j, 1], vertices_coords_2x3, basis_type, i, 0, 0)
                
        H_local = np.zeros((number_of_basis, number_of_basis))
        for k in range(number_of_basis):
            for l in range(number_of_basis):
                integral = np.sum(gauss_weights * lBaG[k, :] * lBaG[l, :])
                H_local[k, l] = integral
        K[np.ix_(element_nodes, element_nodes)] += H_local

    F = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        F[i, :] = cov_g(V_basis[0, :], V_basis[1, :], V_basis[0, i], V_basis[1, i])
        
    M = K @ F @ K
    
    # Solve generalized eigenvalue problem
    lambda_vals, phi = scipy.linalg.eigh(M, K)
    
    sorted_indices = np.argsort(lambda_vals)[::-1]
    lambda_vals, phi = lambda_vals[sorted_indices], phi[:, sorted_indices]
    
    norms = np.linalg.norm(phi, axis=0)
    phi = phi / norms
    
    # Truncate KLE and synthesize samples
    energy_ratio = np.cumsum(lambda_vals) / np.sum(lambda_vals)
    valid_indices = np.where(energy_ratio > thld_KLE)[0]
    dim_TKLE_log_a = valid_indices[0] + 1 if len(valid_indices) > 0 else len(lambda_vals)
    
    rng = np.random.default_rng(seed)
    TKLE_rvs = rng.standard_normal((dim_TKLE_log_a, MC_sample_size))
    truncated_lambda, truncated_phi = lambda_vals[:dim_TKLE_log_a], phi[:, :dim_TKLE_log_a]
    
    weighted_rvs = np.sqrt(truncated_lambda)[:, np.newaxis] * TKLE_rvs
    log_a_sample_KLE_Npts = truncated_phi @ weighted_rvs
    coeff_sample_KLE_Npts_np = np.exp(log_a_sample_KLE_Npts)
    
    coeff_sample_KLE_Npts_torch = torch.from_numpy(coeff_sample_KLE_Npts_np).float()
    
    return coeff_sample_KLE_Npts_torch