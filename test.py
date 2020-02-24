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
BATCH_SIZE = 50
VAL_BATCH_SIZE = 50
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=4)

#try:
device = torch.device('cuda')
#except:
    #device = torch.device('cpu')
output_dim = 10
net = ConvODEUNet(num_filters=32, output_dim=output_dim, time_dependent=True, non_linearity='lrelu', adjoint=True, tol=1e-9)
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
losses = []
val_losses = []
nfe = [[],[],[],[],[],[],[],[],[]]# if TRAIN_UNODE else None
filename = 'best_DD_model_second.pt'
#filename = 'best_DD_model_fourth.pt'
from torch.nn.functional import cosine_similarity

try:
    net = torch.load(filename)
    print("loaded pretrained model")
except:
    print("no pretrained model")

accumulate_batch = 1  # mini-batch size by gradient accumulation
accumulated = 0

def max_num(a, b):
    if a > b:
        return a
    else:
        return b
def min_num(a, b):
    if a > b:
        return b
    else:
        return a

def run(lr, epochs=1):
    zen_coff_GT = [0 for i in range(10)]
    zen_coff_output = [0 for i in range(10)]
    zen_error = [0 for i in range(10)]
    err_dist = [ 0 for i in range(1000)]

    accumulated = 0
    step_size = 100
    count_step = 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    count = 0
    prev_loss = 10000000
    running_loss = 0.0
    total_RMSE = 0.0

    with torch.no_grad():
        for data in valloader:
            count_step = count_step + 1
            if (count_step % step_size) == 0:
                print(count_step)

            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            mseloss = MSE(outputs, labels)
            RMSEloss = torch.sqrt(mseloss)

            outputs = outputs.cpu().clone().numpy()
            labels = labels.cpu().clone().numpy()
            for idx in range(outputs.shape[0]):
                zen_coff_GT = [ zen_coff_GT[i] + labels[idx][i] for i in range(10)]
                zen_coff_output = [zen_coff_output[i] + outputs[idx][i] for i in range(10)]
                zen_error = [ zen_error[i] + (labels[idx][i] - outputs[idx][i])  for i in range(10)]

            if int(mseloss.item() * 1000) <= 1000:
                err_dist[int(mseloss.item() * 1000)] = err_dist[ int(mseloss.item() * 1000)] + 1

            running_loss += loss.item()
            total_RMSE += RMSEloss.item()

            
        zen_coff_GT = [ (i-min(zen_coff_GT)) /  (max(zen_coff_GT) - min(zen_coff_GT) ) for i in zen_coff_GT]
        zen_coff_output = [ (i-min(zen_coff_output) ) / (max(zen_coff_output) - min(zen_coff_output)) for i in zen_coff_output]
        err_dist = [ i/ (len(valloader)/ VAL_BATCH_SIZE) for i in err_dist ]
        zen_error = [ (i / len(valloader)) ** 2 for i in zen_error ]

        import matplotlib.pyplot as plt
        axis_x = [ i/1000 for i in range(1000)]
        plt.figure()
        plt.plot(axis_x[0:100], err_dist[0:100])
        plt.savefig("MSE.png")
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(range(0,10), zen_coff_output, label='output')
        plt.subplot(212)
        plt.plot(range(0,10), zen_coff_GT, label='GT')
        plt.savefig("output_GT_sep_dist.png")
        plt.show()

        plt.figure()
        plt.plot(range(0,10), zen_coff_output, color='r', label ='output')
        plt.plot(range(0,10), zen_coff_GT, color='b', label='GT')
        plt.savefig("output_GT_with_dist.png")
        plt.show()

        plt.figure()
        plt.plot(range(0,10), zen_error, color='r', label ='err')
        plt.savefig("output_GT_with_error.png")
        plt.show()
                            
lr = 1e-3
#epochs = 600 - len(losses)
#lr = 1e-6
epochs = 1
run(lr, epochs)
