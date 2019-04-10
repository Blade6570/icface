import torch
from collections import OrderedDict
#from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
#import itertools
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision
import os
from torch import autograd
import copy
from natsort import natsorted, ns

class Estimate(nn.Module):
    def __init__(self):
        super(Estimate, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, stride=1, padding=1)
         )
        
    def forward(self, x):
        
        x = F.tanh(self.features(x))
        return x
    
class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'
    def _compute_loss_smooth(self, mat):
                return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
                   torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.B=opt.batchSize

        self.netG = torch.nn.DataParallel(networks.define_G(opt.input_nc+20, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids),device_ids=self.gpu_ids)
        
        self.netGN = torch.nn.DataParallel(networks.define_G(opt.input_nc+20, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids),device_ids=self.gpu_ids) 

        self.I_E=Estimate().cuda()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            
            self.netDA = torch.nn.DataParallel(networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids),device_ids=self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netGN, 'GN', opt.which_epoch)
            self.load_network(self.I_E, 'I_E', opt.which_epoch)
            
            if self.isTrain:
                self.load_network(self.netDA, 'DA', opt.which_epoch)

#        self.generator_test = copy.deepcopy(self.netG)
#        self.generator_testN = copy.deepcopy(self.netGN)
        if self.isTrain:
#            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1_F = torch.nn.L1Loss(size_average=False)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionCE = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            
            self.optimizer_I = torch.optim.Adam(self.I_E.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999)) #0.1 in other program

            self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr, alpha=0.99, eps=1e-8)#torch.optim.Adam(self.netG.parameters(), #                                                lr=opt.lr, betas=(0, 0.999),amsgrad=False,weight_decay=0)
            self.optimizer_GN =torch.optim.RMSprop(self.netGN.parameters(), lr=opt.lr, alpha=0.99, eps=1e-8)# torch.optim.Adam(self.netGN.parameters(), #torch.optim.RMSprop(self.netGN.parameters(), lr=opt.lr)#

            self.optimizer_DA = torch.optim.RMSprop(self.netDA.parameters(), lr=opt.lr, alpha=0.99, eps=1e-8)#torch.optim.Adam(self.netDA.parameters(), #torch.optim.RMSprop(self.netDA.parameters(), lr=opt.lr)#
                                           
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_GN)

            self.optimizers.append(self.optimizer_DA)
            self.optimizers.append(self.optimizer_I)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netDA)
        print('-----------------------------------------------')

    def set_input(self, input):

        input_A = input['A'] 

        P_B = input['PB']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)

            P_B = P_B.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A

        self.P_B = P_B

        self.image_paths = input['A_paths']# if AtoB else 'B_paths']

                      
    def test(self):

        self.netG.eval()
        self.netGN.eval()
        self.I_E.eval()
#        pdb.set_trace()
        desti=os.path.dirname(self.image_paths[0])+'/'+self.opt.results_dir

        if not os.path.isdir(desti):
                os.makedirs(desti)     
    
                for i in range(0,self.P_B.size(1)):
                      
                    self.param_B = self.P_B[0,i,:].unsqueeze(0)
                    self.param_B=self.param_B.view(-1,20).float()
  
                    self.real_A=self.input_A[:,0:3,:,:]
                    self.real_A.requires_grad=True

                    I_p=self.I_E(self.real_A)

#                    self.param_A=self.param_A.view(-1,20).float()
                    self.AUN = self.param_B.view(self.param_B.size(0), self.param_B.size(1), 1, 1).expand(
                               self.param_B.size(0), self.param_B.size(1), 128, 128)/100000000
 
                    self.AUN[:,0:3] = 0.5 #

                    self.fake_B = self.netGN(torch.cat([I_p,self.AUN],dim=1))
                    #####################ORIGINAL######################################################
                    

                    AUR = self.param_B.view(self.param_B.size(0), self.param_B.size(1), 1, 1).expand(
                          self.param_B.size(0), self.param_B.size(1), 128, 128)
                    ###################################################################################

                    I_f=self.I_E(self.fake_B)

                    self.fake_B_recon=self.netG(torch.cat([I_f.data,AUR],dim=1))

                    torchvision.utils.save_image((self.fake_B_recon*0.5)+0.5,desti+'/'+str(i)+'_re'+'.png')
 
#                   
                #put your video_path    
                os.system("ffmpeg -r 25 -i ./new_crop/results_video/%01d_re.png -vcodec mpeg4 -y movie.mp4")
                os.system('rm -r ./new_crop/results_video/')

    def get_image_paths(self):
        return self.image_paths
#        return self.ref_path
    def _compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

        
