import random
import pickle
#import cv2
import PIL
import torch
import numpy as np
import torchvision
import scipy.ndimage

#cv2.setNumThreads(0)

class GLaSDataLoader(object):
    def __init__(self, patch_size, dataset_repeat=1, images=np.arange(0, 70), masks = np.arange(0, 70), validation=False, in_dim=3, out_dim=8, out_class=1, Image_fname ='Zernik_images/', Mask_fname ='Zernik_label/'):
        self.image_fname = Image_fname #'Zernik_images/'
        self.mask_fname = Mask_fname #'Zernik_label/'
        self.images = images
        self.masks = masks
        self.out_dim = 8
        self.out_class = 1
        self.in_dim = 3
        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation
        '''
        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            #RandomRotationWithMask(45, resample=False, expand=False, center=None),
            #ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        '''

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.img_open(image, mask)
        return image, mask

    def index_to_filename(self, index):

        image = self.image_fname + str(self.images[index])# + '.TIFF'
        mask = self.mask_fname + str(self.masks[index])# + '.label'
        return image, mask

    def img_open(self, image, mask):
        image = np.load(image)
        file = open(mask, 'r')
        mask =  str(file.readline())[:-2].split(',')
        file.close()
        mask = [float(i) for i in mask]
        image = image.astype(float)
        mask = np.array(mask)
        image = torch.from_numpy(np.array(image)).float()
        mask = torch.from_numpy(np.array(mask)).float()

        return image, mask
    
    def add_noise(self, patch, std = 0.05, mean = 0):
        patch = patch + (torch.randn(patch.shape[0], patch.shape[1], patch.shape[2])*std + mean )
        return patch

    def __len__(self):
        return len(self.images) * self.repeat
