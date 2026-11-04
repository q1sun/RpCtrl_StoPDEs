import numpy as np
import os
import torch
import time
import pickle
import datetime
import h5py
from scipy.sparse import csc_matrix
from scipy.io import loadmat

def load_stiffness_matrices_from_v73(filepath, variable_name):
    """Load v7.3 MATLAB sparse matrices using h5py."""
    matrices = []
    with h5py.File(filepath, 'r') as f:
        if variable_name not in f:
            raise KeyError(f"Variable '{variable_name}' not found in {filepath}")
            
        matrices_refs = f[variable_name][:]
        for ref in matrices_refs.flatten():
            matrix_group = f[ref] 
            if 'MATLAB_sparse' not in matrix_group.attrs:
                continue
            
            data = matrix_group['data'][:]
            ir = matrix_group['ir'][:] 
            jc = matrix_group['jc'][:] 
            shape_attr = matrix_group.attrs['MATLAB_sparse']
            
            if hasattr(shape_attr, '__len__') and len(shape_attr) == 2:
                shape = (int(shape_attr[0]), int(shape_attr[1]))
            else:
                shape = None 
            
            matrices.append(csc_matrix((data, ir, jc), shape=shape))
    return matrices

def main():
    since = time.time()

    # --- Configuration ---
    total_train_size = 5000
    total_test_size = 1000
    output_dir_unsup = 'Data_unsupervised' 
    
    # --- Paths ---
    matlab_A_train_path = 'Data/train_data_KLE_normal/A_matrices_train.mat'
    matlab_A_test_path = 'Data/test_data_KLE_normal/A_matrices_test.mat'
    matlab_tensor_train_path = 'Data/train_data_KLE_normal/Train_Data_KLE_Normal_Tensor.mat'
    matlab_tensor_test_path = 'Data/test_data_KLE_normal/Test_Data_KLE_Normal_Tensor.mat'

    os.makedirs(output_dir_unsup, exist_ok=True)
    print(f"--- Processing Unsupervised Data (Train: {total_train_size}, Test: {total_test_size}) ---")

    print("1. Loading and saving MATLAB stiffness matrices...")
    train_matrices = load_stiffness_matrices_from_v73(matlab_A_train_path, 'train_stiff_matrices')
    test_matrices = load_stiffness_matrices_from_v73(matlab_A_test_path, 'test_stiff_matrices')

    with open(os.path.join(output_dir_unsup, 'train_matrices.pkl'), 'wb') as f: 
        pickle.dump(train_matrices[:total_train_size], f)
    with open(os.path.join(output_dir_unsup, 'test_matrices.pkl'), 'wb') as f: 
        pickle.dump(test_matrices[:total_test_size], f)

    print("2. Loading and saving PyTorch tensors...")
    train_mat = loadmat(matlab_tensor_train_path)
    train_a = train_mat['train_data_inputs_tensor'][:, :, :total_train_size].astype(np.float32)
    train_a_tensors = torch.from_numpy(train_a).permute(2, 0, 1).unsqueeze(1)
    
    test_mat = loadmat(matlab_tensor_test_path)
    test_a = test_mat['test_data_inputs_tensor'][:, :, :total_test_size].astype(np.float32)
    test_a_tensors = torch.from_numpy(test_a).permute(2, 0, 1).unsqueeze(1)

    torch.save(train_a_tensors, os.path.join(output_dir_unsup, 'train_data.pt'))
    torch.save(test_a_tensors, os.path.join(output_dir_unsup, 'test_data.pt'))

    time_elapsed = time.time() - since
    print('\n' + '='*60)
    print(f'Data processing completed in {str(datetime.timedelta(seconds=time_elapsed))}')
    print(f'Files saved to: {output_dir_unsup}/')
    print('='*60)

if __name__ == '__main__':
    main()