# -*- coding: utf-8 -*-
"""Prcoceeing data of face

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from   torch.autograd import Variable
from   skimage import io, transform
from   torch.utils.data import Dataset,DataLoader
from   PIL import Image
from   torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop,Resize,Normalize,ColorJitter
from   torch.utils.data import Dataset,DataLoader
import os
import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt



csv_file="/media/ll/ll/dataset/CelebA/Anno/list_landmarks_celeba.txt"
root_dir="/media/ll/ll/dataset/CelebA/Img/img_align_celeba"


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def hr_transform(crop_size):
    occurrence_color    =random.randint(0,1)#发生色彩改变的数据增强的概率

    if occurrence_color>0.7:
        color_a=random.randint(0,1)
        color_b=random.randint(0,1)
        color_c=random.randint(0,1)
        color_d=random.randint(0,1)
        return Compose([
            ColorJitter(color_a, color_b,color_c, color_d),
            Resize(crop_size,Image.BICUBIC),
            ToTensor()
            ])
    else:
        return Compose([
            Resize(crop_size,Image.BICUBIC),
            ToTensor()
        ])
def lr_transform(crop_size):
    return Compose([
        ToPILImage(),
        Resize(crop_size,Image.BICUBIC),
        ToTensor()
    ])

def data_argument(inp_image,keypoints):

    random_angle=random.randint(-30,30)
    keypoints=keypoints.numpy()
    one_row=[1,1,1,1,1]
    # center_x = round(inp_image.shape[1]/2)
    # center_y = round(inp_image.shape[0]/2)

    new_points = [[], [], []]
    for i in range(len(keypoints)):
        if i%2==0:
            new_points[0].append(keypoints[i])
        else:
            new_points[1].append(keypoints[i])
            new_points[2].append(1)
    new_points = np.array(new_points)
    # print(type(inp_image))
    cv_image = cv2.cvtColor(np.asarray(inp_image),cv2.COLOR_RGB2BGR)
    # print('angle:', random_angle)
    RotateMatrix = cv2.getRotationMatrix2D(center=(cv_image.shape[1]/2, cv_image.shape[0]/2), angle=random_angle, scale=1)
    Roted_img    = cv2.warpAffine(cv_image, RotateMatrix, (cv_image.shape[0]*2, cv_image.shape[1]*2))
    Roted_img=Roted_img[0:218,0:178]
    out_label = np.dot(np.array(RotateMatrix), new_points).reshape(10)
    Roted_img = Image.fromarray(cv2.cvtColor(Roted_img,cv2.COLOR_BGR2RGB))
    
    # print(out_label)
    # exit()
    return  Roted_img,out_label








class FaceLandmarksDataset(Dataset):
    def __init__(self,img_dir,img_txt,):
        img_list=[]
        img_labels=[]
        fp=open(img_txt,"r")
        for line in fp.readlines()[2:]:
            #import pdb
            #pdb.set_trace()
            img_list.append(line.split()[0])
            img_landmarks_single=[]
            for value in line.split()[1:]:
                img_landmarks_single.append(value)
            img_labels.append(img_landmarks_single)

        self.imgs_list = [os.path.join(img_dir,file) for file in img_list ]
        self.labels=img_labels
        self.hr_transform = hr_transform((160,160))
        self.lr_transform = lr_transform((20,20))
    def __getitem__(self,index):
        # print("img name:", self.imgs_list[index])
        occurrence_transform=random.randint(0,1)#发生数据增强的概率
        label_orig = torch.from_numpy(np.array(self.labels[index],dtype=np.int64))
        #print(type(label_orig))
        img_orig=Image.open(self.imgs_list[index])
        if occurrence_transform>0.5:
	        img_roted,label_roted=data_argument(img_orig,label_orig)
	        label_roted=torch.from_numpy(label_roted)
	        hr_image=self.hr_transform(img_roted)
	        lr_image=self.lr_transform(hr_image)
	        #print("happen:",hr_image.size(),lr_image.size(),label_roted.size())
        else:
	        hr_image=self.hr_transform(img_orig)
	        lr_image=self.lr_transform(hr_image)
	        label_roted=torch.DoubleTensor(label_orig.numpy())
	        #print("unhappen:",hr_image.size(),lr_image.size(),label_roted.size())
        return hr_image,lr_image,label_roted
    def __len__(self):
        return len(self.imgs_list)


def _putGaussianMap(center, crop_size_y=80, crop_size_x=80, stride=1, sigma=5):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        center[0]=int(80/178*center[0])
        center[1]=int(80/178*center[0])
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(int(grid_y))]
        x_range = [i for i in range(int(grid_x))]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        # import pdb
        # pdb.set_trace()
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap =np.array(np.exp(-exponent))

        return heatmap

def _putGaussianMaps(keypoints, crop_size_y=160, crop_size_x=160, stride=1, sigma=5):
    """

    :param keypoints: (15,2)
    :param crop_size_y: int
    :param crop_size_x: int
    :param stride: int
    :param sigma: float
    :return:
    """
    all_keypoints = keypoints

    point_num = all_keypoints.shape[0]
    heatmaps_this_img = []
    for k in range(point_num):
        # import pdb
        # pdb.set_trace()
        heatmap = _putGaussianMap(all_keypoints[k],crop_size_y,crop_size_x,stride,sigma)
        heatmap = heatmap[np.newaxis,...]
        heatmaps_this_img.append(heatmap)
    heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
    return heatmaps_this_img
# dataset=FaceLandmarksDataset(root_dir,csv_file)
# Loader=DataLoader(dataset,num_workers=1,batch_size=8)

def OutGaussianMaps(keypoints):
    point=[]
    heatmaps=[]
    for i in range(8):
        point=keypoints[i].view(5,2)
        heatmaps.append(_putGaussianMaps(point))
    heatmaps=np.array(heatmaps)
    
    return heatmaps
# for i ,data in enumerate(Loader):
#     hr,lr,label=data
#     maps=OutGaussianMaps(label)
#     import pdb
#     pdb.set_trace()
#     print(maps[1].shape)



# if __name__ == '__main__':
#     root="/media/ll/ll/dataset/CelebA/Img/img_align_celeba"
#     txt="/media/ll/ll/dataset/CelebA/Anno/list_landmarks_align_celeba.txt"
#     training_dataset=FaceLandmarksDataset(img_dir=root,img_txt=txt)
#     # file = open(txt, 'r')
#     # for line in file:
#     #     print(line)
#     #     import ipdb
#     #     ipdb.set_trace()
#     training_loader=DataLoader(training_dataset,batch_size=8,shuffle=False, num_workers=int(1))
#     for i,data in enumerate(training_loader):
#             high_res_real,low_res,GT_label=data
#             print(GT_label)
