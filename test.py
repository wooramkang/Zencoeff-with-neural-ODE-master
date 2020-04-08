import os
import glob
import random
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data
import torch.nn as nn
import PIL
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook
from models import ConvODEUNet, ConvResUNet, ODEBlock, Unet
from dataloader import GLaSDataLoader

torch.manual_seed(0)
val_img_idr = os.listdir('/project/NANOSCOPY/Submit/Submit/image_integ_val/')
val_mask_idr = os.listdir('/project/NANOSCOPY/Submit/Submit/Zen_integ_val/')
output_idr = '/project/NANOSCOPY/Submit/Submit/Output/'
output_image_idr = '/project/NANOSCOPY/Submit/Submit/Output_image/'
valset = GLaSDataLoader((25, 25), dataset_repeat=1, images=val_img_idr, masks=val_mask_idr ,validation=True, Image_fname ='/project/NANOSCOPY/Submit/Submit/image_integ_val/',
                        Mask_fname ='/project/NANOSCOPY/Submit/Submit/Zen_integ_val/')
VAL_BATCH_SIZE = 1024
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=4)

x_valset = GLaSDataLoader((25, 25), dataset_repeat=1, images=val_img_idr, masks=val_mask_idr ,validation=True, Image_fname ='/project/NANOSCOPY/Submit/Submit/image_integ_val/',
                        Mask_fname ='/project/NANOSCOPY/Submit/Submit/Zen_integ_val/', noise =False)
VAL_BATCH_SIZE = 1024
XNoise_valloader = torch.utils.data.DataLoader(x_valset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=4)

#try:
device = torch.device('cuda')
#except:
    #device = torch.device('cpu')
output_dim = 27
net = ConvODEUNet(num_filters=16, output_dim=output_dim, time_dependent=True, non_linearity='lrelu', adjoint=True, tol=1e-5)
net.to(device)

for m in net.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(net))
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

criterion = torch.nn.BCEWithLogitsLoss()
val_criterion = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
RMSE = RMSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
torch.backends.cudnn.benchmark = True
losses = []
val_losses = []
nfe = [[],[],[],[],[],[],[],[],[]]# if TRAIN_UNODE else None

i = 1
tfboard_path = 'runs/new_zernik_coff_val_' + str(i)

from torch.nn.functional import cosine_similarity
filename = 'best_retriaval_model.pt'

try:
    net = torch.load(filename)
    print("loaded pretrained model")
except:
    print("no pretrained model")

accumulate_batch = 1  # mini-batch size by gradient accumulation
accumulated = 0


zen_coff_GT = [0 for i in range(output_dim)]
zen_coff_output = [0 for i in range(output_dim)]
zen_error = [0 for i in range(output_dim)]
err_dist = [ 0 for i in range(10000)]
x_zen_coff_GT = [0 for i in range(output_dim)]
x_zen_coff_output = [0 for i in range(output_dim)]
x_zen_error = [0 for i in range(output_dim)]
x_err_dist = [ 0 for i in range(10000)]
output_GT_diff = []
x_output_GT_diff = []
output_images = []
x_output_images = []
accumulated = 0
step_size = 10
count_step = 0

count = 0
running_loss = 0.0
total_RMSE = 0.0

lr = 1e-3
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

with torch.no_grad():
    for data in XNoise_valloader:
        count_step = count_step + 1
        if (count_step % step_size) == 0:
            print(count_step)

        inputs, labels = data[0].cuda(), data[1].cuda()
        #noise_in = data[3].cuda()
        outputs = net(inputs)
        MSEloss = MSE (outputs, labels)
        RMSEloss = RMSE(outputs, labels)
        outputs = outputs.cpu().clone().numpy()        
        GT = inputs.cpu().clone().numpy()        
        
        for i in range(len(data[2])):
            #zen_output = open(output_idr + str(data[2][i]), 'w')
            #for j in outputs[i]:
            #    zen_output.write(str(j)+',')
            #zen_output.close()  
            output_image = np.load(output_image_idr + data[2][i].split('.')[0] + '.npy', allow_pickle=True)                  
            #print(output_image)
            output_images.append(output_image)
            output_GT_diff.append(  (np.abs( ( np.sum(GT[i]) - np.sum(output_image) )/( 25 * 25 * 5 ) ))**2 )

        #output_images = np.array(output_images)
        #output_GT_diff.append( (np.sum(GT) - np.sum(output_images))/(outputs.shape[0] * 25 * 25 * 5 ) )

        labels = labels.cpu().clone().numpy()
        diff = [0 for i in range(output_dim)]
        for idx in range(outputs.shape[0]):
            x_zen_coff_GT = [ x_zen_coff_GT[i] + abs(labels[idx][i]/outputs.shape[0]) for i in range(output_dim)]
            x_zen_coff_output = [x_zen_coff_output[i] + abs(outputs[idx][i]/outputs.shape[0]) for i in range(output_dim)]
            x_zen_error = [ x_zen_error[i] + np.abs((labels[idx][i] - outputs[idx][i])/outputs.shape[0])  for i in range(output_dim)]            
            diff = [diff[i] + np.abs((labels[idx][i] - outputs[idx][i])/outputs.shape[0]) for i in range(output_dim)]
            #for i in range(output_dim):
                #print(labels[idx][i])
                #print(outputs[idx][i])
                #print(np.abs((labels[idx][i] - outputs[idx][i])))

        diff = np.array(diff)
        diff_sum = 0
        for i in diff:
            diff_sum = diff_sum + i
        diff_sum = diff_sum / diff.shape[0]
        print("diff")
        print(diff_sum)
        x_err_dist[int(diff_sum * 1000)] = err_dist[ int(diff_sum * 1000)] + 1 

    for data in valloader:
        count_step = count_step + 1
        if (count_step % step_size) == 0:
            print(count_step)

        inputs, labels = data[0].cuda(), data[1].cuda()
        outputs = net(inputs)
        MSEloss = MSE (outputs, labels)
        RMSEloss = RMSE(outputs, labels)
        outputs = outputs.cpu().clone().numpy()        
        GT = inputs.cpu().clone().numpy()        
        
        for i in range(len(data[2])):
            #zen_output = open(output_idr + str(data[2][i]), 'w')
            #for j in outputs[i]:
            #    zen_output.write(str(j)+',')
            #zen_output.close()  
            output_image = np.load(output_image_idr + data[2][i].split('.')[0] + '.npy', allow_pickle=True)                  
            #print(output_image)
            x_output_images.append(output_image)
            x_output_GT_diff.append(  (np.abs( ( np.sum(GT[i]) - np.sum(output_image) )/( 25 * 25 * 5 ) ))**2 )

        #x_output_images = np.array(output_images)
        
        labels = labels.cpu().clone().numpy()
        diff = [0 for i in range(output_dim)]
        for idx in range(outputs.shape[0]):
            zen_coff_GT = [ zen_coff_GT[i] + abs(labels[idx][i]/outputs.shape[0]) for i in range(output_dim)]
            zen_coff_output = [zen_coff_output[i] + abs(outputs[idx][i]/outputs.shape[0]) for i in range(output_dim)]
            zen_error = [ zen_error[i] + np.abs((labels[idx][i] - outputs[idx][i])/outputs.shape[0])  for i in range(output_dim)]            
            diff = [diff[i] + np.abs((labels[idx][i] - outputs[idx][i])/outputs.shape[0]) for i in range(output_dim)]
            #for i in range(output_dim):
                #print(labels[idx][i])
                #print(outputs[idx][i])
                #print(np.abs((labels[idx][i] - outputs[idx][i])))

        diff = np.array(diff)
        diff_sum = 0
        for i in diff:
            diff_sum = diff_sum + i
        diff_sum = diff_sum / diff.shape[0]
        print("diff")
        print(diff_sum)
        err_dist[int(diff_sum * 1000)] = err_dist[ int(diff_sum * 1000)] + 1
    
    zen_error = [ i / len(valloader)  for i in zen_error ]
    zen_coff_output = [ i / len(valloader)  for i in zen_coff_output ]      
    zen_coff_GT = [ i / len(valloader)  for i in zen_coff_GT ]              
    err_dist = [ i/ len(valloader) for i in err_dist ]
    err_dist = np.array(err_dist)
    
    x_zen_error = [ i / len(valloader)  for i in x_zen_error ]
    x_zen_coff_output = [ i / len(valloader)  for i in x_zen_coff_output ]      
    x_zen_coff_GT = [ i / len(valloader)  for i in x_zen_coff_GT ]              
    x_err_dist = [ i/ len(valloader) for i in x_err_dist ]
    x_err_dist = np.array(err_dist)

    poi_dist = [0 for i in range(1000)]
    for s in range(10000):
        s = (np.random.poisson(16)/ 3125) #* output_dim
        poi_dist[int(s*1000)] = poi_dist[int(s*1000)] + 1

    poi_dist = np.array(poi_dist)

    axis_x = [ i/1000 for i in range(1000)]
    plt.figure()    
    plt.plot(axis_x[0:300], x_err_dist[0:300], color='b', label ='error without noise')
    plt.plot(axis_x[0:300], err_dist[0:300], color='r', label ='error with noise')
    plt.savefig("output_error_dist.png")
    plt.show()

    plt.figure()    
    plt.plot(axis_x[0:300], err_dist[0:300], color='b', label ='error with noise')
    plt.plot(axis_x[0:300], poi_dist[0:300], color='r', label ='noise')
    plt.savefig("output_error_poi_dist.png")
    plt.show()

    plt.figure()
    plt.plot(axis_x[0:300], err_dist[0:300], color='r', label ='loss with noise')
    plt.savefig("output_only_error_dist.png")
    plt.show()
    
    plt.figure()
    plt.plot(axis_x[0:300], err_dist[0:300], color='r', label ='noise')
    plt.savefig("output_only_noise_dist.png")
    plt.show()

    plt.figure()
    plt.plot(axis_x[0:300], x_err_dist[0:300], color='r', label ='loss with noise')
    plt.savefig("output_only_error_withoutNoise_dist.png")
    plt.show()
    Zen_range = range(0,output_dim)

    plt.figure()
    plt.plot(Zen_range, zen_coff_output, color='r', label ='output')
    plt.plot(Zen_range, zen_coff_GT, color='b', label='GT')
    plt.savefig("output_GT_with_dist.png")
    plt.show()

    plt.figure()
    plt.plot(Zen_range, zen_error, color='r', label ='err')
    plt.savefig("output_GT_with_loss.png")
    plt.show()

    plt.figure()
    plt.plot(range( len(output_GT_diff) ), output_GT_diff, color='r', label ='err')
    plt.savefig("output_GT_image_with_loss.png")
    plt.show()

    hist_diff = [0 for i in range(1000)]
    for i in output_GT_diff:
        delta = int(i * 1000)
        hist_diff[delta] = hist_diff[delta] + 1
    
    x_hist_diff =[]    
    for i in x_output_GT_diff:
        delta = int(i * 1000)
        x_hist_diff[delta] = x_hist_diff[delta] + 1

    plt.figure()
    axis_x = [ i/1000 for i in range(1000)]
    plt.plot(axis_x[0:100], hist_diff[0:100], color='r', label ='err')     
    plt.plot(axis_x[0:100], poi_dist[0:100], color='b', label ='poi_err')
    plt.savefig("output_GT_image_with_loss_hist.png")
    plt.show()
    plt.figure()
    
    plt.plot(axis_x[0:100], hist_diff[0:100], color='r', label ='err')     
    plt.plot(axis_x[0:100], x_hist_diff[0:100], color='b', label ='poi_err')
    plt.savefig("output_wihtoutorwith_noise.png")
    plt.show()