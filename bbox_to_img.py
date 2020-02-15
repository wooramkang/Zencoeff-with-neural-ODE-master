import numpy as np
import cv2
import os
import sys
import pickle

for r, d, f in os.walk("frames/"):
    for file in f:
        im = cv2.imread(os.path.join(r, file))
        im = cv2.resize(im, (512, 512))
        mask = np.zeros( (512, 512, 1), np.uint8)
        target_mask = np.zeros( (512, 512, 8), np.uint8)

        print("gt_bbox/"+file[:-3]+"txt")
        bbox_list = open("gt_bbox/"+file[:-3]+"txt", 'r')
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        for bbox in bbox_list.readlines():
            target_class = int(bbox[0])
            #print(target_class)
            bbox = bbox.split(' ')[1:]
            #print(bbox)
            #print(im.shape)
            #print(im.shape[0])
            bbox[0] = int(float(bbox[0]) * im.shape[0])
            bbox[2] = int(float(bbox[2])* im.shape[0])
            bbox[1] = int(float(bbox[1])* im.shape[1])
            bbox[3] = int(float(bbox[3])* im.shape[1])
            t_mask = np.zeros( (512, 512, 1), np.uint8)
            #print(bbox)
            cv2.rectangle(mask,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(255),-1)
            cv2.rectangle(t_mask,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(255),-1)
            
            grab_mask = np.zeros( (512, 512, 1), np.uint8)
            grab_mask[t_mask == 255] = 1

            preproc_mask, bgdModel, fgdModel = cv2.grabCut(im, grab_mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
            preproc_mask = np.where((preproc_mask==2)|(preproc_mask==0),0, 1).astype('uint8')
        
            temp = np.ones( (im.shape[0], im.shape[1], 1), np.uint8)
            temp = temp * preproc_mask[:,:]#,np.newaxis]
            #cv2.imshow('scrab', temp*255)#*255)
            mask_nonzero = np.where(target_mask == 0, 1, 0 )
            #target_mask= target_mask + temp * mask_nonzero[:, :]
            target_mask[:,:,target_class] = target_mask[:,:,target_class] + temp[:,:,0]
            #cv2.imshow('scrab_temp', target_mask[:,:,target_class]*255)#*255)

            grab_im = im * preproc_mask[:,:]#,np.newaxis]
            #print(grab_im.shape)
            #cv2.imshow('grab', grab_im)
            
            #cv2.imwrite(str(i)+'.png', im)
            #cv2.waitKey(5000)
        pickle.dump(target_mask, open("maskout/"+file[:-3]+"pkl", "wb") )
        #cv2.imshow('grab_img', target_mask*255)
        #cv2.imshow('Features', mask)
        #cv2.imshow('IMG', im)
        #cv2.imwrite(str(i)+'.png', im)
        #cv2.waitKey(5000)


cv2.destroyAllWindows()
