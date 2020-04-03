import os
import glob
import random
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data

import PIL
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook
from models import ConvODEUNet, ConvResUNet, ODEBlock, Unet
from dataloader import GLaSDataLoader

torch.manual_seed(0)
train_img_idr = os.listdir('/project/NANOSCOPY/Submit/Submit/image_integ/')
train_mask_idr = os.listdir('/project/NANOSCOPY/Submit/Submit/Zen_integ/')
val_img_idr = os.listdir('/project/NANOSCOPY/Submit/Submit/image_integ_val/')
val_mask_idr = os.listdir('/project/NANOSCOPY/Submit/Submit/Zen_integ_val/')

trainset = GLaSDataLoader((25, 25), dataset_repeat=1, images=train_img_idr, masks=train_mask_idr, Image_fname ='/project/NANOSCOPY/Submit/Submit/image_integ/', 
                        Mask_fname ='/project/NANOSCOPY/Submit/Submit/Zen_integ/')
valset = GLaSDataLoader((25, 25), dataset_repeat=1, images=val_img_idr, masks=val_mask_idr ,validation=True, Image_fname ='/project/NANOSCOPY/Submit/Submit/image_integ_val/',
                        Mask_fname ='/project/NANOSCOPY/Submit/Submit/Zen_integ_val/')
BATCH_SIZE = 3 # The Auther of the paper neural ODE said training with batch is not sure 
VAL_BATCH_SIZE = 1000
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=2)

i = 1
tfboard_path = 'runs/new_zernik_coff_' + str(i)
writer = SummaryWriter(tfboard_path)

#try:
device = torch.device('cuda')
#except:
    #device = torch.device('cpu')
output_dim = 28 # ex) 30, 28, 10, ... maximum of NOLL index 
net = ConvODEUNet(num_filters=32, output_dim=output_dim, time_dependent=True, non_linearity='lrelu', adjoint=True, tol=1e-7)
net.to(device)

for m in net.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(net))

criterion = torch.nn.BCEWithLogitsLoss()
val_criterion = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
torch.backends.cudnn.benchmark = True

filename = 'best_retriaval_model.pt'
from torch.nn.functional import cosine_similarity

try:
    net = torch.load(filename)
    print("loaded pretrained model")
except:
    print("no pretrained model")

accumulate_batch = 1  # mini-batch size by gradient accumulation
accumulated = 0

def run(lr, epochs=100):
    accumulated = 0
    step_size = 2000
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    count = 0
    prev_loss = 10000000

    for epoch in range(epochs):
        # training loop with gradient accumulation
        e_count = 0    
        running_loss = 0.0
        optimizer.zero_grad()
        
        for data in tqdm(trainloader):
            e_count = e_count + 1
            count = count + 1
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels) / accumulate_batch
            #RMSEloss = torch.sqrt(MSE(outputs, labels)) #/ accumulate_batch                       
            #RMSEloss.backward()
            loss.backward()
            accumulated += 1
            if accumulated == accumulate_batch:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

            running_loss += loss.item() * accumulate_batch
            #running_loss += RMSEloss.item() * accumulate_batch
            if (count % step_size) == 0:
                print((running_loss / e_count)/BATCH_SIZE)
                writer.add_scalar('training_loss', (running_loss / e_count)/BATCH_SIZE, (count/step_size) )

                running_loss = 0.0
                e_count = 0
                cos_all = 0

                # validation loop
                with torch.no_grad():
                    running_loss = 0.0
                    total_RMSE = 0.0

                    for data in valloader:
                        #print(data[1].shape)
                        inputs, labels = data[0].cuda(), data[1].cuda()
                        outputs = net(inputs)
                        #print(outputs.cpu().clone().numpy().shape)
                        loss = val_criterion(outputs, labels)
                        RMSEloss = torch.sqrt(MSE(outputs, labels))                        
                        outputs = outputs.cpu().clone().numpy()
                        outputs = (outputs + 1) / 2
                        outputs = torch.from_numpy(outputs).float()
                        labels = labels.cpu().clone().numpy()
                        labels = (labels + 1) / 2
                        labels = torch.from_numpy(labels).float()
                        cos_similarity = cosine_similarity(outputs, labels, dim=1)
                        cos_similarity = cos_similarity.cpu().clone().numpy()
                        #print(cos_similarity)
                        cos_similarity = np.sum(np.absolute(cos_similarity))/ VAL_BATCH_SIZE
                        #print(cos_similarity)
                        cos_all = cos_all + cos_similarity
                        running_loss += loss.item()
                        total_RMSE += RMSEloss.item()

                    cos_all = cos_all / len(valloader)
                    print(cos_all)
                    #val_losses.append(running_loss / len(valloader))
                    writer.add_scalar('validation_RMSEloss', total_RMSE / len(valloader), (count/step_size) )
                    writer.add_scalar('validation_loss', running_loss / len(valloader), (count/step_size) )
                    writer.add_scalar('cosine_similarity', cos_all, (count/step_size))

                    if prev_loss >= (total_RMSE / len(valloader)):
                        print("model saved")
                        torch.save(net, filename)
                        prev_loss = (running_loss / len(valloader))
                
lr = 1e-3
#epochs = 600 - len(losses)
#lr = 1e-6
epochs = 100
run(lr, epochs)
