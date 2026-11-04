import numpy as np

def generate_mesh_linear_quadratic_element_py(h):
    """
    Generates linear and quadratic meshes for a 2D unit square domain.
    """
    # ==========================================================================
    # 1. Generate Linear Element Mesh
    # ==========================================================================
    left, right, bottom, top = 0.0, 1.0, 0.0, 1.0
    
    N1_linear = int((right - left) / h[0])
    N2_linear = int((top - bottom) / h[1])
    tnp_linear = (N1_linear + 1) * (N2_linear + 1)
    
    V_linear = np.zeros((2, tnp_linear))
    T_linear = np.zeros((3, 2 * N1_linear * N2_linear), dtype=int)
    Q_linear = np.zeros((N1_linear + 1, N2_linear + 1), dtype=int)

    for j in range(1, tnp_linear + 1):
        if j % (N2_linear + 1) == 0:
            V_linear[0, j - 1] = left + (j / (N2_linear + 1) - 1) * h[0]
            V_linear[1, j - 1] = top
        else:
            V_linear[0, j - 1] = left + np.floor(j / (N2_linear + 1)) * h[0]
            V_linear[1, j - 1] = bottom + (j % (N2_linear + 1) - 1) * h[1]
            
    for i in range(1, N1_linear + 2):
        for j in range(1, N2_linear + 2):
            Q_linear[i - 1, j - 1] = (i - 1) * (N2_linear + 1) + j

    for n in range(1, N1_linear * N2_linear + 1):
        if n % N2_linear == 0:
            row = N2_linear
            column = n // N2_linear
        else:
            row = n % N2_linear
            column = n // N2_linear + 1
        
        T_linear[0, 2 * n - 2] = Q_linear[column - 1, row - 1]
        T_linear[1, 2 * n - 2] = Q_linear[column, row - 1]
        T_linear[2, 2 * n - 2] = Q_linear[column - 1, row]
        
        T_linear[0, 2 * n - 1] = Q_linear[column - 1, row]
        T_linear[1, 2 * n - 1] = Q_linear[column, row - 1]
        T_linear[2, 2 * n - 1] = Q_linear[column, row]

    T_linear = T_linear - 1
    
    nbn_linear = 2 * (N1_linear + N2_linear)
    BN_linear_Dirichlet = np.zeros(nbn_linear, dtype=int)
    
    for k in range(1, N1_linear + 1):
        BN_linear_Dirichlet[k - 1] = (k - 1) * (N2_linear + 1) + 1
    k_offset = N1_linear
    for k in range(1, N2_linear + 1):
        BN_linear_Dirichlet[k_offset + k - 1] = N1_linear * (N2_linear + 1) + k
    k_offset = N1_linear + N2_linear
    for k in range(1, N1_linear + 1):
        BN_linear_Dirichlet[k_offset + k - 1] = (N1_linear + 2 - k) * (N2_linear + 1)
    k_offset = 2 * N1_linear + N2_linear
    for k in range(1, N2_linear + 1):
        BN_linear_Dirichlet[k_offset + k - 1] = (N2_linear + 2 - k)
        
    BN_linear_Dirichlet = BN_linear_Dirichlet - 1

    nbe = 2 * (N1_linear + N2_linear)
    BE_linear_Dirichlet = np.zeros((4, nbe), dtype=int)
    BE_linear_Dirichlet[2, :] = -1 
    
    for k in range(1, N1_linear + 1):
        BE_linear_Dirichlet[3, k - 1] = (k - 1) * 2 * N2_linear + 1
        BE_linear_Dirichlet[0, k - 1] = (k - 1) * (N2_linear + 1) + 1
        BE_linear_Dirichlet[1, k - 1] = k * (N2_linear + 1) + 1
    for k in range(N1_linear + 1, N1_linear + N2_linear + 1):
        BE_linear_Dirichlet[3, k - 1] = (N1_linear - 1) * 2 * N2_linear + 2 * (k - N1_linear)
        BE_linear_Dirichlet[0, k - 1] = N1_linear * (N2_linear + 1) + k - N1_linear
        BE_linear_Dirichlet[1, k - 1] = N1_linear * (N2_linear + 1) + k - N1_linear + 1
    for k in range(N1_linear + N2_linear + 1, 2 * N1_linear + N2_linear + 1):
        BE_linear_Dirichlet[3, k - 1] = (2 * N1_linear + N2_linear + 1 - k) * 2 * N2_linear
        BE_linear_Dirichlet[0, k - 1] = (2 * N1_linear + N2_linear + 2 - k) * (N2_linear + 1)
        BE_linear_Dirichlet[1, k - 1] = (2 * N1_linear + N2_linear + 1 - k) * (N2_linear + 1)
    for k in range(2 * N1_linear + N2_linear + 1, nbe + 1):
        BE_linear_Dirichlet[3, k - 1] = 2 * (2 * N1_linear + 2 * N2_linear + 1 - k) - 1
        BE_linear_Dirichlet[0, k - 1] = 2 * N1_linear + 2 * N2_linear + 2 - k
        BE_linear_Dirichlet[1, k - 1] = 2 * N1_linear + 2 * N2_linear + 1 - k
    
    BE_linear_Dirichlet[0:2, :] = BE_linear_Dirichlet[0:2, :] - 1
    BE_linear_Dirichlet[3, :] = BE_linear_Dirichlet[3, :] - 1

    # ==========================================================================
    # 2. Generate Quadratic Element Mesh
    # ==========================================================================
    N1_quadratic_coarse, N2_quadratic_coarse = N1_linear, N2_linear
    dh = (h[0]/2, h[1]/2)
    dN1, dN2 = N1_quadratic_coarse * 2, N2_quadratic_coarse * 2
    tnp_quadratic = (dN1 + 1) * (dN2 + 1)

    V_quadratic = np.zeros((2, tnp_quadratic))
    T_quadratic = np.zeros((6, 2 * N1_quadratic_coarse * N2_quadratic_coarse), dtype=int)
    Q_quadratic = np.zeros((dN1 + 1, dN2 + 1), dtype=int)

    for j in range(1, tnp_quadratic + 1):
        if j % (dN2 + 1) == 0:
            V_quadratic[0, j - 1] = left + (j / (dN2 + 1) - 1) * dh[0]
            V_quadratic[1, j - 1] = top
        else:
            V_quadratic[0, j - 1] = left + np.floor(j / (dN2 + 1)) * dh[0]
            V_quadratic[1, j - 1] = bottom + (j % (dN2 + 1) - 1) * dh[1]

    for i in range(1, dN1 + 2):
        for j in range(1, dN2 + 2):
            Q_quadratic[i - 1, j - 1] = (i - 1) * (dN2 + 1) + j
            
    for n in range(1, N1_quadratic_coarse * N2_quadratic_coarse + 1):
        if n % N2_quadratic_coarse == 0:
            row = N2_quadratic_coarse
            column = n // N2_quadratic_coarse
        else:
            row = n % N2_quadratic_coarse
            column = n // N2_quadratic_coarse + 1
        
        T_quadratic[0, 2*n-2] = Q_quadratic[2*column-2, 2*row-2]
        T_quadratic[1, 2*n-2] = Q_quadratic[2*column, 2*row-2]
        T_quadratic[2, 2*n-2] = Q_quadratic[2*column-2, 2*row]
        T_quadratic[3, 2*n-2] = Q_quadratic[2*column-1, 2*row-2]
        T_quadratic[4, 2*n-2] = Q_quadratic[2*column-1, 2*row-1]
        T_quadratic[5, 2*n-2] = Q_quadratic[2*column-2, 2*row-1]
        
        T_quadratic[0, 2*n-1] = Q_quadratic[2*column-2, 2*row]
        T_quadratic[1, 2*n-1] = Q_quadratic[2*column, 2*row-2]
        T_quadratic[2, 2*n-1] = Q_quadratic[2*column, 2*row]
        T_quadratic[3, 2*n-1] = Q_quadratic[2*column-1, 2*row-1]
        T_quadratic[4, 2*n-1] = Q_quadratic[2*column, 2*row-1]
        T_quadratic[5, 2*n-1] = Q_quadratic[2*column-1, 2*row]
    T_quadratic = T_quadratic - 1

    N1_bn_q, N2_bn_q = dN1, dN2
    nbn_quadratic = 2 * (N1_bn_q + N2_bn_q)
    BN_quadratic_Dirichlet = np.zeros(nbn_quadratic, dtype=int)
    
    for k in range(1, N1_bn_q + 1): 
        BN_quadratic_Dirichlet[k-1] = (k-1)*(N2_bn_q+1)+1
    for k in range(N1_bn_q + 1, N1_bn_q + N2_bn_q + 1): 
        BN_quadratic_Dirichlet[k-1] = N1_bn_q*(N2_bn_q+1)+k-N1_bn_q
    for k in range(N1_bn_q + N2_bn_q + 1, 2*N1_bn_q + N2_bn_q + 1): 
        BN_quadratic_Dirichlet[k-1] = (2*N1_bn_q+N2_bn_q+2-k)*(N2_bn_q+1)
    for k in range(2*N1_bn_q + N2_bn_q + 1, nbn_quadratic + 1): 
        BN_quadratic_Dirichlet[k-1] = 2*N1_bn_q+2*N2_bn_q+2-k
    BN_quadratic_Dirichlet = BN_quadratic_Dirichlet - 1

    nbe_q = 2*(N1_bn_q+N2_bn_q)
    BE_quadratic_Dirichlet=np.zeros((4,nbe_q), dtype=int)
    BE_quadratic_Dirichlet[2,:]=-1
    
    for k in range(1, N1_bn_q+1):
        BE_quadratic_Dirichlet[3,k-1]=(k-1)*2*N2_bn_q+1
        BE_quadratic_Dirichlet[0,k-1]=(k-1)*(N2_bn_q+1)+1
        BE_quadratic_Dirichlet[1,k-1]=k*(N2_bn_q+1)+1
    for k in range(N1_bn_q+1,N1_bn_q+N2_bn_q+1):
        BE_quadratic_Dirichlet[3,k-1]=(N1_bn_q-1)*2*N2_bn_q+2*(k-N1_bn_q)
        BE_quadratic_Dirichlet[0,k-1]=N1_bn_q*(N2_bn_q+1)+k-N1_bn_q
        BE_quadratic_Dirichlet[1,k-1]=N1_bn_q*(N2_bn_q+1)+k-N1_bn_q+1
    for k in range(N1_bn_q+N2_bn_q+1,2*N1_bn_q+N2_bn_q+1):
        BE_quadratic_Dirichlet[3,k-1]=(2*N1_bn_q+N2_bn_q+1-k)*2*N2_bn_q
        BE_quadratic_Dirichlet[0,k-1]=(2*N1_bn_q+N2_bn_q+2-k)*(N2_bn_q+1)
        BE_quadratic_Dirichlet[1,k-1]=(2*N1_bn_q+N2_bn_q+1-k)*(N2_bn_q+1)
    for k in range(2*N1_bn_q+N2_bn_q+1,nbe_q+1):
        BE_quadratic_Dirichlet[3,k-1]=2*(2*N1_bn_q+2*N2_bn_q+1-k)-1
        BE_quadratic_Dirichlet[0,k-1]=2*N1_bn_q+2*N2_bn_q+2-k
        BE_quadratic_Dirichlet[1,k-1]=2*N1_bn_q+2*N2_bn_q+1-k
        
    BE_quadratic_Dirichlet[[0,1,3], :] = BE_quadratic_Dirichlet[[0,1,3], :] - 1

    results = {
        'V_linear': V_linear,
        'T_linear': T_linear,
        'BN_linear_Dirichlet': BN_linear_Dirichlet,
        'BE_linear_Dirichlet': BE_linear_Dirichlet,
        'V_quadratic': V_quadratic,
        'T_quadratic': T_quadratic,
        'BN_quadratic_Dirichlet': BN_quadratic_Dirichlet,
        'BE_quadratic_Dirichlet': BE_quadratic_Dirichlet
    }
    return results