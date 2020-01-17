#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:30:12 2019

@author:Soumya
"""
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
from PIL import Image
import pandas
#from natsort import natsorted, ns
#import glob
#import numpy as np
#import sys
#sys.path.append('/home/esa/Downloads/face-alignment')

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot)
        self.ref_path=opt.which_ref
        self.csv_path=opt.csv_path
 
    def __getitem__(self, index):
   
        SI_path= self.ref_path
        
        df2 = pandas.read_csv(self.csv_path)
        a=list(range(296,299))+ list(range(679,696))#+ list(range(299,435))
        refdf = df2[df2.columns[a]]
        
        f=0
        A1 = Image.open(SI_path).convert('RGB')            
        A1 = A1.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A1 = transforms.ToTensor()(A1)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A1)
         
        for i in range(0,len(refdf)):
            
            paramA=torch.tensor(refdf.values[i]).view(1,-1) 
            paramA[0,0:3]=(paramA[0,0:3]-(-0.70))/1.4
            paramA[0,3:20]=paramA[0,3:20]/5
        
            if f==0:
                P1=paramA
            else:               
                P1=torch.cat([P1,paramA],dim=0)               
            f=f+1
        
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)

        return {'A': A, 'PB': P1,
                'A_paths': SI_path}#SAB_path[0]

    def __len__(self):
        return len(self.ref_path)

    def name(self):
        return 'AlignedDataset'
