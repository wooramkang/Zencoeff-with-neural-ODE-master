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
from inference_utils import inference_image, postprocess
from models import ConvODEUNet, ConvResUNet, ODEBlock, Unet
from dataloader import GLaSDataLoader
from train_utils import plot_losses

torch.manual_seed(0)
train_img_idr = os.listdir('Zernik_images/')
train_mask_idr = os.listdir('Zernik_label/')
val_img_idr = os.listdir('val_Zernik_images/')
val_mask_idr = os.listdir('val_Zernik_label/')

trainset = GLaSDataLoader((25, 25), dataset_repeat=1, images=train_img_idr, masks=train_mask_idr, Image_fname ='Zernik_images/', Mask_fname ='Zernik_label/')
valset = GLaSDataLoader((25, 25), dataset_repeat=1, images=val_img_idr, masks=val_mask_idr ,validation=True, Image_fname ='val_Zernik_images/', Mask_fname ='val_Zernik_label/')
BATCH_SIZE = 12
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
writer = SummaryWriter('runs/zernik_coff')

#try:
device = torch.device('cuda')
#except:
    #device = torch.device('cpu')

net = ConvODEUNet(num_filters=16, output_dim=10, time_dependent=True, non_linearity='lrelu', adjoint=True, tol=1e-3)
net.to(device)

for m in net.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(net)

criterion = torch.nn.BCEWithLogitsLoss()
val_criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
torch.backends.cudnn.benchmark = True
losses = []
val_losses = []
nfe = [[],[],[],[],[],[],[],[],[]]# if TRAIN_UNODE else None
filename = 'best_DD_model.pt'

try:
    #net = torch.load(filename)
    print("loaded pretrained model")
except:
    print("no pretrained model")

accumulate_batch = 1  # mini-batch size by gradient accumulation
accumulated = 0


def run(lr=1e-3, epochs=100):
    accumulated = 0
    step_size = 500
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
            loss.backward()
            accumulated += 1
            if accumulated == accumulate_batch:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

            running_loss += loss.item() * accumulate_batch

            if (count % step_size) == 0:
                writer.add_scalar('training_loss', (running_loss / e_count)/BATCH_SIZE, (count/step_size) )

                if prev_loss >= ((running_loss / e_count)/BATCH_SIZE):
                    print("model saved")
                    torch.save(net, filename)
                    prev_loss = ((running_loss / e_count)/BATCH_SIZE)

                running_loss = 0.0
                e_count = 0

                # validation loop
                with torch.no_grad():
                    running_loss = 0.0
                    for data in valloader:
                        inputs, labels = data[0].cuda(), data[1].cuda()
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()

                    val_losses.append(running_loss / len(valloader))

                    writer.add_scalar('validation_loss', running_loss / len(valloader), (count/step_size) )

                    #if prev_loss >= (running_loss / len(valloader)):
                        #print("model saved")
                        #torch.save(net, filename)
                        #prev_loss = (running_loss / len(valloader))
                
        '''
        # test result loop
        index = 0
        for data in valloader:
            fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(4*3,3*5))
            index = index + 1
            ax[0, 0].set_title('Image')
            ax[0, 1].set_title('Ground-truth')
            ax[0, 2].set_title('Unode')

            for col in range(3):
                for row in range(5):
                    image = data[0]
                    gt = data[1]
                    with torch.no_grad():
                        result, input_image = inference_image(net, image)#, shouldpad=TRAIN_UNET)
        '''
lr = 1e-3 
run(lr, 600 - len(losses))


