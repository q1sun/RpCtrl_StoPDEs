'''
Training script via supervised learning for 
robust-pathwise control of elliptic PDEs with KLE coefficients
(Native PyTorch FNO Version)
'''
from Data.data_loader_py import preprocess_traindata, preprocess_testdata
from torch.utils.data import Dataset, DataLoader
from Models.Robust_Pathwise_Control.models_py import fno
from Models.Robust_Pathwise_Control.utils_py import helper
from torch.optim.lr_scheduler import MultiStepLR

import torch
import numpy as np
import scipy.io as io
import torch.nn as nn
import os
import time
import datetime
import argparse

parser = argparse.ArgumentParser(description='PyTorch RpCtrl KLE SL Training')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='Checkpoints/Model_RobPathCtrl_KLE_Normal/simulation_0', type=str, metavar='PATH', 
                    help='path to save checkpoint')
# model outputs
parser.add_argument('-o', '--output', default='Checkpoints/RobPathCtrl_KLE_Normal/simulation_0', type=str, metavar='PATH', 
                    help='path to save model output')
args = parser.parse_args()

##############################################################################################
print('*', '*' * 40, '*')
print('Data-driven Robust-pathwise Control via Supervised Learning (Native FNO)', "\n")
print("pytorch version", torch.__version__, "\n")
print('*', '*' * 40, '*', "\n", "\n")
##############################################################################################

##############################################################################################
## hyperparameter configuration
batch_size = 100
input_chnls = 1 
output_chnls = 1
num_epochs = 100
milestones = 60, 80
##############################################################################################

##############################################################################################
print('*', '-' * 45, '*')
print('===> preparing training and testing datasets ...')
print('*', '-' * 45, '*')

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
        return coef, ctrl
    
train_set = TrainDataset(input_chnls, output_chnls, transform = None)

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
        return coef, ctrl
    
test_set = TestDataset(input_chnls, output_chnls, transform = None)

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),
}

print('TRAINING:')
print('traindata_coef = {}'.format(train_set.sto_coefs.shape))
print('traindata_ctrl = {}'.format(train_set.opt_ctrls.shape), "\n")

print('TESTING:')
print('testdata_coef = {}'.format(test_set.sto_coefs.shape))
print('testdata_ctrl = {}'.format(test_set.opt_ctrls.shape))
print('*', '-' * 40, '*', "\n", "\n")
##############################################################################################

##############################################################################################
print('*', '-' * 45, '*')
print('===> creating training and testing functions ...')

def train_epoch(epoch, model, criterion, optimizer, device):
    model.train()
    epoch_loss = 0                                                   
    for sto_coefs, opt_ctrls in dataloaders['train']:
        sto_coefs = sto_coefs.to(device)
        opt_ctrls = opt_ctrls.to(device)            
        
        outputs = model(sto_coefs)
        loss = criterion(outputs, opt_ctrls)                
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()        
        epoch_loss += loss.item()
    return epoch_loss

def test_epoch(epoch, model, criterion, optimizer, device):
    model.eval()
    epoch_loss = 0                                                   
    with torch.no_grad():
        for sto_coefs, opt_ctrls in dataloaders['test']:
            sto_coefs = sto_coefs.to(device)
            opt_ctrls = opt_ctrls.to(device)            
            
            outputs = model(sto_coefs)
            loss = criterion(outputs, opt_ctrls)                
            
            epoch_loss += loss.item()
    return epoch_loss
##############################################################################################

##############################################################################################
print('*', '-' * 45, '*')
print('===> training neural network ...')

if not os.path.isdir(args.checkpoint):
    helper.mkdir_p(args.checkpoint)

criterion = nn.MSELoss() 

model = fno.FNO2d(
    modes1=12,
    modes2=12,
    width=64,
    in_channels=input_chnls,
    out_channels=output_chnls,
    dropout=0.0,
    img_size=(train_set.sto_coefs.shape[2], train_set.sto_coefs.shape[3]),
    padding=3
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device), "\n")
model = model.to(device)
criterion = criterion.to(device)

logger = helper.Logger(os.path.join(args.checkpoint, 'log.txt'), title='Train-RobPathCtrl-KLE-FNO')
logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss'])
     
train_loss, test_loss = [], []
trainloss_best = 1e10
since = time.time()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))  
        
    trainloss_epoch = train_epoch(epoch, model, criterion, optimizer, device)
    testloss_epoch = test_epoch(epoch, model, criterion, optimizer, device)
    
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
    
    train_avg_loss = trainloss_epoch * batch_size / train_set.sto_coefs.shape[0]
    test_avg_loss = testloss_epoch * batch_size / test_set.sto_coefs.shape[0]
    
    logger.append([optimizer.param_groups[0]['lr'], train_avg_loss, test_avg_loss])
    
    schedular.step()
        
    train_loss.append(train_avg_loss)
    test_loss.append(test_avg_loss)
    
    print('==> Average Loss: Training = {:.4e}'.format(train_avg_loss), ', Testing = {:.4e}'.format(test_avg_loss), "\n")

logger.close()
time_elapsed = time.time() - since
print('Training finished in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################

##############################################################################################
print('*', '-' * 45, '*')
print('===> loading trained model for inference ...')
since = time.time()

checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0),
    'test': DataLoader(test_set, batch_size=test_set.sto_coefs.shape[0], shuffle=False, num_workers=0),
}

model.eval()
with torch.no_grad():
    train_inputs, train_targets = next(iter(dataloaders['train'])) 
    test_inputs, test_targets = next(iter(dataloaders['test']))

    train_inputs = train_inputs.to(device)
    test_inputs = test_inputs.to(device)

    train_predicts = model(train_inputs)  
    test_predicts = model(test_inputs)

print('trainbatch_model_predicts = {}'.format(train_predicts.shape))
print('testdata_model_predicts = {}'.format(test_predicts.shape))

if not os.path.isdir(args.output):
    helper.mkdir_p(args.output)
    
io.savemat(os.path.join(args.output, 'Trainbatch_RobPathCtrl_KLE_Normal_Tensor.mat'), {'trainbatch_rob_path_ctrl_OptCtrl_Tensor':np.array(train_predicts.cpu().detach())})
io.savemat(os.path.join(args.output, 'Testdata_RobPathCtrl_KLE_Normal_Tensor.mat'), {'testdata_rob_path_ctrl_OptCtrl_Tensor':np.array(test_predicts.cpu().detach())})

time_elapsed = time.time() - since
print('Testing finished in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################