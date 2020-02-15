import random
import pickle
#from augmentations import ElasticTransformations, RandomRotationWithMask
#import cv2
import PIL
import torch
import numpy as np
import torchvision
import scipy.ndimage

#cv2.setNumThreads(0)

class GLaSDataLoader(object):
    def __init__(self, patch_size, dataset_repeat=1, images=np.arange(0, 70), masks = np.arange(0, 70), validation=False, in_dim=3, out_dim=8, out_class=1):
        self.image_fname = 'Zernik_images/'
        self.mask_fname = 'Zernik_label/'
        self.images = images
        self.masks = masks
        self.out_dim = 8
        self.out_class = 1
        self.in_dim = 3
        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation

        self.image_mask_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            #RandomRotationWithMask(45, resample=False, expand=False, center=None),
            #ElasticTransformations(2000, 60),
            torchvision.transforms.ToTensor()
        ])
        self.image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            #torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.img_open(image, mask)
        return image, mask
        #image, mask = self.pad_image(image, mask)
        #label, patch = self.apply_data_augmentation(image, mask)
        #label = self.create_eroded_mask(label, mask)
        #patch, label = self.extract_random_region(image, patch, label)
        #return patch, label.float()

    def index_to_filename(self, index):
        """Helper function to retrieve filenames from index"""
        if index == 0:
            index = 1
        if index > 30000:
            index = 30000

        image = self.image_fname + str(index) + '.TIFF'
        mask = self.mask_fname + str(index) + '.label'
        return image, mask

    def img_open(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        image = PIL.Image.open(image)
        #image = image.resize((32, 32))
        file = open(mask, 'r')
        mask =  str(file.readline())[:-2].split(',')
        file.close()
        mask = [float(i) for i in mask]
        
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype(float)
        mask = np.array(mask)
        image = torch.from_numpy(np.array(image)).float()
        mask = torch.from_numpy(np.array(mask)).float()

        return image, mask

    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        if not self.validation:
            pad_h = max(self.patch_size[0] - image.shape[0], 128)
            pad_w = max(self.patch_size[1] - image.shape[1], 128)
        else:
            # we pad more than needed to later do translation augmentation
            pad_h = max((self.patch_size[0] - image.shape[0]) // 2 + 1, 0)
            pad_w = max((self.patch_size[1] - image.shape[1]) // 2 + 1, 0)

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        return padded_image, mask

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        patch = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
        #n_glands = mask.max()
        label = torch.from_numpy(mask.transpose(2, 0, 1)).float() #/ n_glands
        
        if not self.validation:
            
            #patch_label_concat = torch.cat((patch, label[None, :, :].float()))
            #patch_label_concat = self.image_mask_transforms(patch_label_concat)
            #patch, label = patch_label_concat[0:3], np.round(patch_label_concat[3:] * n_glands)
            #patch = self.image_transforms(patch)
            
            noise_idx = np.random.randint(6)
            if (noise_idx % 2) == 0:
                patch = add_noise(patch, std = 0.025)
            
            noise_idx = np.random.randint(6)
            if (noise_idx % 2) == 0:
                label = add_noise(label, std = 0.01)
        else:
            label *= n_glands
        return label, patch
    
    def add_noise(self, patch, std = 0.05, mean = 0):
        patch = patch + (torch.randn(patch.shape[0], patch.shape[1], patch.shape[2])*std + mean )
        return patch

    def create_eroded_mask(self, label, mask):
        """Helper function to create a mask where every gland is eroded"""
        boundaries = torch.zeros(label.shape)
        for i in np.unique(mask):
            if i == 0: continue  # the first label is background
            gland_mask = (label == i).float()
            binarized_mask_border = scipy.ndimage.morphology.binary_erosion(gland_mask,
                                                                            structure=np.ones((13, 13)),
                                                                            border_value=1)

            binarized_mask_border = torch.from_numpy(binarized_mask_border.astype(np.float32))
            boundaries[label == i] = binarized_mask_border[label == i]

        label = (label > 0).float()
        label = torch.stack((boundaries, label))
        return label

    def extract_random_region(self, image, patch, label):
        """Helper function to perform translation data augmentation"""
        if not self.validation:
            loc_y = random.randint(0, image.shape[0] - self.patch_size[0])
            loc_x = random.randint(0, image.shape[1] - self.patch_size[1])
        else:
            loc_y, loc_x = 0, 0

        patch = patch[:, loc_y:loc_y+self.patch_size[0], loc_x:loc_x+self.patch_size[1]]
        label = label[:, loc_y:loc_y+self.patch_size[0], loc_x:loc_x+self.patch_size[1]]
        return patch, label

    def __len__(self):
        return len(self.images) * self.repeat