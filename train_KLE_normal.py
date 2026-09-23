'''
Training script via supervised learning for 
         robust-pathwise control of elliptic PDEs with KLE coefficients
'''

from Data.data_loader_py import preprocess_traindata, preprocess_testdata
from torch.utils.data import Dataset, DataLoader
from Models.Robust_Pathwise_Control.models_py import unet
from Models.Robust_Pathwise_Control.utils_py import helper
from torch.optim.lr_scheduler import MultiStepLR

import torch
import numpy as np
import scipy.io as io

import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
import argparse

parser = argparse.ArgumentParser(description='PyTorch RpCtrl KLE SL Training')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_0', type=str, metavar='PATH', 
                    help='path to save checkpoint (default: Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_0)')
# model outputs
parser.add_argument('-o', '--output', default='Checkpoints/RobPathCtrl_KLE_Normal/simulation_0', type=str, metavar='PATH', 
                    help='path to save model output (default: Checkpoints/RobPathCtrl_KLE_Normal/simulation_0)')
args = parser.parse_args()


##############################################################################################
print('*', '*' * 40, '*')
print('Data-driven Robust-pathwise Control via Supervised Learning', "\n")
print("pytorch version", torch.__version__, "\n")
print('*', '*' * 40, '*', "\n", "\n")
##############################################################################################


##############################################################################################
## hyperparameter configuration
batch_size = 100
input_chnls = 1 
output_chnls = 1
num_epochs = 250
milestones = 120, 200
##############################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> preparing training and testing datasets ...')
print('*', '-' * 45, '*')

# training dataset 
class TrainDataset(Dataset):    
    def __init__(self, input_chnls, output_chanls, transform=None):
        self.sto_coefs, self.opt_ctrls = preprocess_traindata.preprocess_trainset(input_chnls, output_chnls)    
        self.transform = transform
        
    def __len__(self):
        return len(self.sto_coefs)
    
    def __getitem__(self, idx):
        coef = self.sto_coefs[idx]
        ctrl = self.opt_ctrls[idx]
        if self.transform:
            coef = self.transform(coef)

        return [coef, ctrl]
    
train_set = TrainDataset(input_chnls, output_chnls, transform = None)

# testing dataset
class TestDataset(Dataset):    
    def __init__(self, input_chnls, output_chanls, transform=None):
        self.sto_coefs, self.opt_ctrls = preprocess_testdata.preprocess_testset(input_chnls, output_chnls)    
        self.transform = transform
        
    def __len__(self):
        return len(self.sto_coefs)
    
    def __getitem__(self, idx):
        coef = self.sto_coefs[idx]
        ctrl = self.opt_ctrls[idx]
        if self.transform:
            coef = self.transform(coef)

        return [coef, ctrl]
    
test_set = TestDataset(input_chnls, output_chnls, transform = None)

# define dataloader
dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0),
}

# print information associated with training data
print('TRAINING:')
print('traindata_coef (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(train_set.sto_coefs.shape))
print('               with', train_set.sto_coefs.dtype, 'elements defined between {:.4e}'.format(torch.min(train_set.sto_coefs)), 'and {:.4e}'.format(torch.max(train_set.sto_coefs)))
print('traindata_ctrl (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(train_set.opt_ctrls.shape))
print('               with', train_set.opt_ctrls.dtype, 'elements defined between {:.4e}'.format(torch.min(train_set.opt_ctrls)), 'and {:.4e}'.format(torch.max(train_set.opt_ctrls)))

train_inputs, train_targets = next(iter(dataloaders['train'])) # get a batch of training data
print('traindata_batch_input (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(train_inputs.shape))
print('traindata_batch_output (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(train_targets.shape), "\n")

# print information associated with testing data
print('TESTING:')
print('testdata_coef (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(test_set.sto_coefs.shape))
print('              with', test_set.sto_coefs.dtype, 'elements defined between {:.4e}'.format(torch.min(test_set.sto_coefs)), 'and {:.4e}'.format(torch.max(test_set.sto_coefs)))
print('testdata_ctrl (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(test_set.opt_ctrls.shape))
print('              with', test_set.opt_ctrls.dtype, 'elements defined between {:.4e}'.format(torch.min(test_set.opt_ctrls)), 'and {:.4e}'.format(torch.max(test_set.opt_ctrls)))

# test_inputs, test_targets = next(iter(dataloaders['test'])) # get a batch of training data
# print('testdata_batch_input: (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(test_inputs.shape))
# print('testdata_batch_output: (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(test_targets.shape), "\n")
print('*', '-' * 40, '*', "\n", "\n")
##############################################################################################




##############################################################################################
print('*', '-' * 45, '*')
print('===> creating training model ...')
print('*', '-' * 45, '*', "\n", "\n")

def train_epoch(epoch, model, criterion, optimizer, device):
    
    # set model to training mode
    model.train()

    epoch_loss, epoch_batch = 0, 0                                                   
    for sto_coefs, opt_ctrls in dataloaders['train']:
        
        # send inputs, outputs to device
        sto_coefs = sto_coefs.to(device)
        opt_ctrls = opt_ctrls.to(device)            
        
        # forward pass 
        outputs = model(sto_coefs)
        loss = criterion(outputs, opt_ctrls)                
        # zero parameter gradients
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # parameter update
        optimizer.step()        
        
        # integrate loss      
        epoch_loss += loss.item()
        
        # print results
        epoch_batch += sto_coefs.size(0)
        # print('      TRAINING: batch_index = {}/{}'.format(epoch_batch, train_set.sto_coefs.shape[0]), 
        #       'with batch_loss = {:.4e}'.format(loss.item()))        
        
    return epoch_loss
##############################################################################################




##############################################################################################
print('*', '-' * 45, '*')
print('===> creating testing model ...')
print('*', '-' * 45, '*', "\n", "\n")

def test_epoch(epoch, model, criterion, optimizer, device):
    
    # set model to testing mode
    model.eval()

    epoch_loss, epoch_batch = 0, 0                                                   
    for sto_coefs, opt_ctrls in dataloaders['test']:
        
        # send inputs, outputs to device
        sto_coefs = sto_coefs.to(device)
        opt_ctrls = opt_ctrls.to(device)            
        
        # forward pass 
        outputs = model(sto_coefs)
        loss = criterion(outputs, opt_ctrls)                
        
        # integrate loss      
        epoch_loss += loss.item()
        
        # print results
        epoch_batch += sto_coefs.size(0)
        # print('      TESTING: batch_index = {}/{}'.format(epoch_batch, train_set.sto_coefs.shape[0]), 
        #       'with batch_loss = {:.4e}'.format(loss.item()))        
    
    return epoch_loss
##############################################################################################





##############################################################################################
print('*', '-' * 45, '*')
print('===> training neural network ...')

if not os.path.isdir(args.checkpoint):
    helper.mkdir_p(args.checkpoint)

# create model
criterion = nn.MSELoss() 
model = unet.UNet()

# create optimizer and learning rate schedular
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=0.01)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

# load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device), "\n")
model = model.to(device)
criterion = criterion.to(device)

# create log file
logger = helper.Logger(os.path.join(args.checkpoint, 'log.txt'), title='Train-RobPathCtrl-KLE-SL')
logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss'])
     
# train and test 
train_loss, test_loss = [], []
trainloss_best = 1e10
since = time.time()
for epoch in range(num_epochs):
      
    print('Epoch {}/{}'.format(epoch, num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))  
    # print('     ', '-' * 20)
        
    # execute training and testing
    trainloss_epoch = train_epoch(epoch, model, criterion, optimizer, device)
    # print('     ', '-' * 20)
    testloss_epoch = test_epoch(epoch, model, criterion, optimizer, device)
    # print('     ', '-' * 20)
    
    # save current and best models to checkpoint
    is_best = trainloss_epoch < trainloss_best
    if is_best:
        print('==> Saving best model ...')
    trainloss_best = min(trainloss_epoch, trainloss_best)
    helper.save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'trainloss_epoch': trainloss_epoch,
                            'testloss_epoch': testloss_epoch,
                            'trainloss_best': trainloss_best,
                            'optimizer': optimizer.state_dict(),
                           }, is_best, checkpoint=args.checkpoint)   
    # save training process to log file
    logger.append([optimizer.param_groups[0]['lr'], trainloss_epoch * batch_size / train_set.sto_coefs.shape[0], testloss_epoch * batch_size / test_set.sto_coefs.shape[0]])
    
    # adjust learning rate according to predefined schedule
    schedular.step()
        
    # print results
    train_loss.append(trainloss_epoch)
    test_loss.append(testloss_epoch)
    print('==> Batch Loss: Training = {:.4e}'.format(trainloss_epoch), ', Testing = {:.4e}'.format(testloss_epoch), "\n")

logger.close()
time_elapsed = time.time() - since
print('Training finished in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################




##############################################################################################
print('*', '-' * 45, '*')
print('===> loading trained model for inference ...')
since = time.time()

# load trained model
checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

# creat full-batch dataloaders
dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0),
    'test': DataLoader(test_set, batch_size=test_set.sto_coefs.shape[0], shuffle=False, num_workers=0),
}

# load full-batch (training and testing) input data to device
train_inputs, train_targets = next(iter(dataloaders['train']))
test_inputs, test_targets = next(iter(dataloaders['test']))
train_inputs = train_inputs.to(device)
test_inputs = test_inputs.to(device)

# forward pass
train_predicts = model(train_inputs)
test_predicts = model(test_inputs)
print('trainbatch_model_predicts (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(train_predicts.shape))
print('testdata_model_predicts (num_smps, num_chnls, num_Xcoord, num_Ycoord) = {}'.format(test_predicts.shape))

# save results in '.mat' format
if not os.path.isdir(args.output):
    helper.mkdir_p(args.output)
    
io.savemat(os.path.join(args.output, 'Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat'), {'trainbatch_rob_path_ctrl_OptCtrl_Tensor':np.array(train_predicts.cpu().detach())})
io.savemat(os.path.join(args.output, 'Testdata_RobPathCtrl_KLE_Normal_Tensor.mat'), {'testdata_rob_path_ctrl_OptCtrl_Tensor':np.array(test_predicts.cpu().detach())})

time_elapsed = time.time() - since
print('Testing finished in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################



