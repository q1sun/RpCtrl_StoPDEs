import os
import torch
torch.set_default_dtype(torch.float32)

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import datetime
import argparse
import pickle
import scipy.io as io
from joblib import Parallel, delayed

from Models.Robust_Pathwise_Control.models_py.unet import UNet
from Models.Robust_Pathwise_Control.utils_py.helper import Logger, save_checkpoint, mkdir_p
from Models.Robust_Pathwise_Control.fem_solver import (
    assemble_mass_matrix, precompute_fem_constants, get_U_target_nodes,
    solve_for_one_sample, compute_primal_objective, weighted_mse_loss_raw,
    img_to_vec_torch, vec_to_img_torch, unnormalize_f_batch_torch, normalize_f_batch_torch
)
from Data.data_generator_py.generate_mesh_linear_quadratic_element_py import generate_mesh_linear_quadratic_element_py as generate_mesh

# ==============================================================================
# Dataset & Collate
# ==============================================================================
class UnsupervisedKLEDataset(Dataset):    
    def __init__(self, data_path, matrices_path):
        self.sto_coefs = torch.load(data_path).float()
        with open(matrices_path, 'rb') as A:
            self.matrices = pickle.load(A)
        print(f"UnsupervisedKLEDataset: sto_coefs shape: {self.sto_coefs.shape}")
        
    def __len__(self):
        return len(self.sto_coefs)
        
    def __getitem__(self, idx):
        return self.sto_coefs[idx], self.matrices[idx]

def sparse_matrix_collate(batch):
    a_tensors = [item[0] for item in batch]
    A_matrices = [item[1] for item in batch]
    return torch.stack(a_tensors, 0), A_matrices

# ==============================================================================
# Trainer & Tester
# ==============================================================================
def train_epoch_unsupervised(model, optimizer, device, dataloader, beta, fem_data, n_jobs, eta, M_torch, epoch):
    model.train()
    total_J_monitor = 0.0

    for batch_idx, (batch_a, batch_A) in enumerate(dataloader):
        batch_a = batch_a.to(device, non_blocking=True).float()
        
        with torch.no_grad():
            f_k_raw_gpu = unnormalize_f_batch_torch(img_to_vec_torch(model(batch_a)), fem_data)
            
            if epoch == 0 and batch_idx == 0:
                f_k_raw_gpu = torch.zeros_like(f_k_raw_gpu)
                
            batch_f_raw_numpy = f_k_raw_gpu.cpu().numpy()

            results = Parallel(n_jobs=n_jobs, batch_size=max(1, len(batch_f_raw_numpy)//2))(
                delayed(solve_for_one_sample)(batch_A[i], batch_f_raw_numpy[i, :], fem_data)
                for i in range(len(batch_f_raw_numpy))
            )
            batch_u_numpy = np.array([r[0] for r in results])
            batch_xi_numpy = np.array([r[1] for r in results])
            
            batch_xi_vec_np = np.array([x.flatten(order='F') for x in batch_xi_numpy])
            xi_k_raw_gpu = torch.from_numpy(batch_xi_vec_np).float().to(device, non_blocking=True)

            gradient_k = beta * f_k_raw_gpu + xi_k_raw_gpu
            f_target_raw = f_k_raw_gpu - eta * gradient_k
            f_target_raw = f_target_raw * fem_data['boundary_mask']
                
            f_target_norm = normalize_f_batch_torch(f_target_raw, fem_data).detach()

            for i in range(len(batch_f_raw_numpy)):
                total_J_monitor += compute_primal_objective(
                    batch_u_numpy[i], batch_f_raw_numpy[i].astype(np.float64), fem_data, beta
                )

        optimizer.zero_grad(set_to_none=True)
        pred_unsup_vec = img_to_vec_torch(model(batch_a))
        pred_raw = unnormalize_f_batch_torch(pred_unsup_vec, fem_data)
        target_raw = unnormalize_f_batch_torch(f_target_norm, fem_data)
        
        loss_unsup = weighted_mse_loss_raw(pred_raw, target_raw, M_torch)
        loss_unsup.backward()
        optimizer.step()
        
        if batch_idx % 5 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
            
    return total_J_monitor / len(dataloader.dataset)


def test_epoch_unsupervised(model, device, dataloader, beta, fem_data, n_jobs, eta, M_torch):
    model.eval()
    total_J = 0.0

    with torch.no_grad():
        for batch_a, batch_A in dataloader:
            batch_a = batch_a.to(device, non_blocking=True).float()
            f_k_raw_gpu = unnormalize_f_batch_torch(img_to_vec_torch(model(batch_a)), fem_data)
            
            batch_f_raw_numpy = f_k_raw_gpu.cpu().numpy()
            results = Parallel(n_jobs=n_jobs, batch_size=max(1, len(batch_f_raw_numpy)//2))(
                delayed(solve_for_one_sample)(batch_A[i], batch_f_raw_numpy[i, :], fem_data)
                for i in range(len(batch_f_raw_numpy))
            )
            batch_u_numpy = np.array([r[0] for r in results])
            
            for i in range(len(batch_f_raw_numpy)):
                total_J += compute_primal_objective(batch_u_numpy[i], batch_f_raw_numpy[i].astype(np.float64), fem_data, beta)
            
    return total_J / len(dataloader.dataset)

# ==============================================================================
# Main
# ==============================================================================
def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description='Physics-Informed Unsupervised Training')
    parser.add_argument('-c', '--checkpoint', default='Checkpoints/Model_RobPathCtrl_KLE_Unsupervised/simulation_0', type=str, metavar='PATH', 
                        help='path to save checkpoint')
    parser.add_argument('-o', '--output', default='Checkpoints/RobPathCtrl_KLE_Unsupervised/simulation_0', type=str, metavar='PATH', 
                        help='path to save model output')
    args = parser.parse_args()

    output_dir = args.output if args.output is not None else args.checkpoint
    if not os.path.isdir(output_dir): mkdir_p(output_dir)

    h = (1/32, 1/32)
    batch_size = 250                
    num_epochs = 200
    learning_rate = 4e-3
    beta = 1e-10
    milestones = [160, 180]
    eta = 90.0
    num_parallel_jobs = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"Starting Unsupervised Training | Device: {device}")
    print("="*60, "\n")

    mesh_data = generate_mesh(h=h)
    U_target = lambda x, y: np.sin(2 * np.pi * x) * (np.cos(2 * np.pi * y) - 1)
    
    fem_data = {
        'V_basis': mesh_data['V_linear'], 'T_basis': mesh_data['T_linear'],
        'BN_Dirichlet': mesh_data['BN_linear_Dirichlet'],
        'num_nodes': mesh_data['V_linear'].shape[1], 'num_elements': mesh_data['T_linear'].shape[1],
        'num_gauss_pts': 7, 'U_target': U_target,
        'nx': int(1/h[0] + 1), 'ny': int(1/h[1] + 1),
    }

    print("===> Loading normalization constants from .mat file...")
    tensor_train_mat = io.loadmat('Data/train_data_KLE_normal/Train_Data_KLE_Normal_Tensor.mat')
    
    fem_data['min_f'] = float(tensor_train_mat['min_path_ctrls'].flatten()[0])
    fem_data['max_f'] = float(tensor_train_mat['max_path_ctrls'].flatten()[0])
    fem_data['mean_f_gpu'] = torch.from_numpy(tensor_train_mat['mean_path_ctrls'].flatten()).float().to(device)

    fem_data = precompute_fem_constants(fem_data)
    fem_data['bn_indices_tensor'] = torch.from_numpy(fem_data['BN_Dirichlet']).long().to(device)

    mask_np = np.ones(fem_data['num_nodes'], dtype=np.float32)
    mask_np[fem_data['BN_Dirichlet']] = 0.0
    fem_data['boundary_mask'] = torch.from_numpy(mask_np).to(device)

    fem_data['mass_matrix'] = assemble_mass_matrix(fem_data)
    M_sp = fem_data['mass_matrix'].tocoo()
    indices = torch.LongTensor(np.vstack((M_sp.row, M_sp.col)))
    values = torch.FloatTensor(M_sp.data)
    M_torch = torch.sparse_coo_tensor(indices, values, torch.Size(M_sp.shape)).to(device)

    get_U_target_nodes(fem_data)

    train_dataset_unsup = UnsupervisedKLEDataset('Data_unsupervised/train_data.pt', 'Data_unsupervised/train_matrices.pkl')
    test_dataset_unsup = UnsupervisedKLEDataset('Data_unsupervised/test_data.pt', 'Data_unsupervised/test_matrices.pkl')

    dataloaders = {
        'train_unsup': DataLoader(train_dataset_unsup, batch_size=batch_size, shuffle=True, collate_fn=sparse_matrix_collate, num_workers=0, pin_memory=True),
        'test_unsup': DataLoader(test_dataset_unsup, batch_size=batch_size, shuffle=False, collate_fn=sparse_matrix_collate, num_workers=0, pin_memory=True)
    }

    model = UNet().to(device).float()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='Unsupervised_Training')
    logger.set_names(['LR', 'Train J', 'Test J'])

    print('===> Starting training...')
    best_test_J = 1e10

    init_test_J = test_epoch_unsupervised(model, device, dataloaders['test_unsup'], beta, fem_data, num_parallel_jobs, eta, M_torch)
    init_train_J = test_epoch_unsupervised(model, device, dataloaders['train_unsup'], beta, fem_data, num_parallel_jobs, eta, M_torch)
    
    logger.append([optimizer.param_groups[0]['lr'], init_train_J, init_test_J])
    print(f'==> Initial State: Train J={init_train_J:.4e}, Test J={init_test_J:.4e}\n')

    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs} | LR = {optimizer.param_groups[0]["lr"]:.1e}')
        
        train_J = train_epoch_unsupervised(model, optimizer, device, dataloaders['train_unsup'], beta, fem_data, num_parallel_jobs, eta, M_torch, epoch)
        test_J = test_epoch_unsupervised(model, device, dataloaders['test_unsup'], beta, fem_data, num_parallel_jobs, eta, M_torch)
        
        scheduler.step()
        is_best = test_J < best_test_J
        if is_best:
            best_test_J = test_J
            print(f"  Saving Best Model (Test J: {best_test_J:.4e})...")
        
        save_checkpoint({
            'epoch': epoch + 1, 'state_dict': model.state_dict(),
            'train_J': train_J, 'test_J': test_J, 'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        logger.append([optimizer.param_groups[0]['lr'], train_J, test_J])
        print(f'==> Epoch Results: Train J={train_J:.4e} | Test J={test_J:.4e}\n')
    
    logger.close()

    print('===> Generating final inference results...')
    best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"  Loaded best model from epoch {checkpoint['epoch']}.")
    model.eval()

    eval_train_loader = DataLoader(train_dataset_unsup, batch_size=batch_size, shuffle=False, collate_fn=sparse_matrix_collate, num_workers=0)
    
    all_train_predicts = []
    with torch.no_grad():
        for batch_a, _ in eval_train_loader:
            batch_a = batch_a.to(device).float()
            pred_raw = unnormalize_f_batch_torch(img_to_vec_torch(model(batch_a)), fem_data)
            pred_raw = pred_raw * fem_data['boundary_mask']
            pred_norm = normalize_f_batch_torch(pred_raw, fem_data)
            all_train_predicts.append(vec_to_img_torch(pred_norm, fem_data).cpu())
            
    train_predicts_tensor = torch.cat(all_train_predicts, dim=0)
    io.savemat(os.path.join(output_dir, 'Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat'), 
               {'trainbatch_rob_path_ctrl_OptCtrl_Tensor': train_predicts_tensor.numpy()})

    all_test_predicts = []
    with torch.no_grad():
        for batch_a, _ in dataloaders['test_unsup']:
            batch_a = batch_a.to(device).float()
            pred_raw = unnormalize_f_batch_torch(img_to_vec_torch(model(batch_a)), fem_data)
            pred_raw = pred_raw * fem_data['boundary_mask']
            pred_norm = normalize_f_batch_torch(pred_raw, fem_data)
            all_test_predicts.append(vec_to_img_torch(pred_norm, fem_data).cpu())
            
    test_predicts_tensor = torch.cat(all_test_predicts, dim=0)
    io.savemat(os.path.join(output_dir, 'Testdata_RobPathCtrl_KLE_Normal_Tensor.mat'), 
               {'testdata_rob_path_ctrl_OptCtrl_Tensor': test_predicts_tensor.numpy()})
               
    print(f"===> Train/Test results saved to {output_dir} with boundary zero.")

    time_elapsed = time.time() - since
    print('='*60)
    print(f'Training complete! Total time: {str(datetime.timedelta(seconds=time_elapsed))}')
    print('='*60)

if __name__ == '__main__':
    main()