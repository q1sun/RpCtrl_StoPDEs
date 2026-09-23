from scipy.io import loadmat
import numpy as np
import torch

def preprocess_testset(input_chnls, output_chnls):
    testset = loadmat('Data/test_data_KLE_normal/Test_Data_KLE_Normal_Tensor.mat')


    # modify input channels for testing data (sto-coefs) if necessary
    testset_sto_coefs = testset['test_data_inputs_tensor'][ np.newaxis, ... ]
    for idx_chnl in range(input_chnls-1): # tunable hyperparameter, e.g., input_chnls = 3 
        testset_sto_coefs = np.concatenate( (testset_sto_coefs, testset['test_data_inputs_tensor'][None]), axis = 0 )         
    # data type: float64 to float32
    testset_sto_coefs = np.float32(testset_sto_coefs)   
    # permute CHWN to NCHW 
    testset_sto_coefs = torch.from_numpy(testset_sto_coefs).permute(3,0,1,2)


    # modify output channels for testing data (opt-ctrls) if necessary
    testset_opt_ctrls = testset['test_data_targets_tensor'][ np.newaxis, ... ]
    for idx_chnl in range(output_chnls-1): # tunable hyperparameter, e.g., output_chnls = 6
        testset_opt_ctrls = np.concatenate( (testset_opt_ctrls, testset['test_data_targets_tensor'][None]), axis = 0 )             
    # data type: float64 to float32
    testset_opt_ctrls = np.float32(testset_opt_ctrls)
    # permute CHWN to NCHW 
    testset_opt_ctrls = torch.from_numpy(testset_opt_ctrls).permute(3,0,1,2)
    
    
    return testset_sto_coefs, testset_opt_ctrls