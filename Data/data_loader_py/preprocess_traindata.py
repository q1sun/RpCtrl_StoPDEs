from scipy.io import loadmat
import numpy as np
import torch

def preprocess_trainset(input_chnls, output_chnls):
    trainset = loadmat('Data/train_data_KLE_normal/Train_Data_KLE_Normal_Tensor.mat')


    # modify input channels for training data (sto-coefs) if necessary
    trainset_sto_coefs = trainset['train_data_inputs_tensor'][ np.newaxis, ... ]
    for idx_chnl in range(input_chnls-1): # tunable hyperparameter, e.g., input_chnls = 3 
        trainset_sto_coefs = np.concatenate( (trainset_sto_coefs, trainset['train_data_inputs_tensor'][None]), axis = 0 )         
    # data type: float64 to float32
    trainset_sto_coefs = np.float32(trainset_sto_coefs)   
    # permute CHWN to NCHW 
    trainset_sto_coefs = torch.from_numpy(trainset_sto_coefs).permute(3,0,1,2)


    # modify output channels for training data (opt-ctrls) if necessary
    trainset_opt_ctrls = trainset['train_data_targets_tensor'][ np.newaxis, ... ]
    for idx_chnl in range(output_chnls-1): # tunable hyperparameter, e.g., output_chnls = 6
        trainset_opt_ctrls = np.concatenate( (trainset_opt_ctrls, trainset['train_data_targets_tensor'][None]), axis = 0 )             
    # data type: float64 to float32
    trainset_opt_ctrls = np.float32(trainset_opt_ctrls)
    # permute CHWN to NCHW 
    trainset_opt_ctrls = torch.from_numpy(trainset_opt_ctrls).permute(3,0,1,2)
    
    
    return trainset_sto_coefs, trainset_opt_ctrls