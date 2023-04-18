import os
import json
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import time


# Utility methods to calculate convolution and transpose convolution sizes
def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

# Loss func for autoencoder
def loss_function(recon_x, x):
    # MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return MSE

# Autoencoder class
class autoencoder(nn.Module):
    def __init__(self, fc_hidden1=256, drop_p=0.3, embed_dim=32):
        super(autoencoder, self).__init__()

        self.fc_hidden1, self.embed_dim = fc_hidden1, embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k3, self.k5 = (3, 3), (5,5)      # 2d kernal size
        self.s2 = (2, 2)      # 2d strides
        self.pd0, self.pd1 = (0, 0), (1, 1)  # 2d padding
        
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=self.k5, stride=self.s2, padding=self.pd1),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.k3, stride=self.s2, padding=self.pd1),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.k3, stride=self.s2, padding=self.pd0),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        
        # Lin encoder
        self.encoder_lin = nn.Sequential(
            nn.Linear(7 * 7 * 32, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden1, self.embed_dim),
            #nn.ReLU(inplace=True),

        )
            
        # Lin decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(self.embed_dim, self.fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden1, 7 * 7 * 32),
            nn.ReLU(inplace=True),
        )
         
        # unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 7, 7))

        # Decoder
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.k3, stride=self.s2, padding=self.pd0),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.k3, stride=self.s2, padding=self.pd1),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k5, stride=self.s2, padding=self.pd1),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten(x)
        
        x = self.encoder_lin(x)
        
        return x
        

    def decode(self, z):
        x = self.decoder_lin(z)
        
        x = self.unflatten(x)
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear')
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst, z


# AE training function for 1 epoch
def train(log_interval, model, device, train_loader, optimizer, epoch, save_model_path):
    
    start_time = time.time()
    # set model as training mode
    model.train()

    losses = []
    all_y, all_z = [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, X  in enumerate(train_loader):

        # distribute data to device
        X = X.to(device)
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z = model(X)  # AE
        loss = loss_function(X_reconst, X)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        all_z.extend(z.data.cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
        
        #end_time = time.time()
        #print('time tot: ', "{:.8f}".format(end_time - start_time))
        
    all_z = np.stack(all_z, axis=0)
   
    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))
    end_time = time.time()
    print('epoch train time: ', "{:.8f}".format(end_time - start_time))
        
    return X.data.cpu().numpy(), all_z, losses



# AE validation function

def evaluate(log_interval, model, device, optimizer, test_loader):
    start_time = time.time()
    
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_z = []
    all_X_reconst = []
    
    
    N_count = 0   # counting total evalued samples

    with torch.no_grad():
        for batch_idx, X in enumerate(test_loader):
            N_count += X.size(0)
            
            # distribute data to device
            X = X.to(device)
            
            # Use the model to get embedding and reconstruction
            X_reconst, z = model(X)

            loss = loss_function(X_reconst, X)
            test_loss += loss.item()  # sum up batch loss

            all_z.extend(z.data.cpu().numpy())
            #all_X_reconst.extend(X_reconst.data.cpu().numpy())

            
            # Show information
            if (batch_idx + 1) % log_interval == 0:
                print('Eval: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    N_count, len(test_loader.dataset), 100. * (batch_idx + 1) / len(test_loader), loss.item()))

    test_loss /= len(test_loader.dataset)
    all_z = np.stack(all_z, axis=0)
    all_X_reconst = np.stack(all_X_reconst, axis=0)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), test_loss))
    end_time = time.time()
    print('eval time: ', "{:.8f}".format(end_time - start_time))
    
    return X.data.cpu().numpy(), all_z, all_X_reconst, test_loss
