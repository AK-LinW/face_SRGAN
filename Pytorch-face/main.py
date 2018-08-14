# -*- coding: utf-8 -*-
import torch
from   torch.nn import Upsample
from   torch.autograd import Variable
from   PIL import Image
from   tensorboard_logger import configure,log_value
from   torch.autograd import variable
from   torch.utils.data import Dataset,DataLoader
from   prcoceData import  FaceLandmarksDataset,OutGaussianMaps
from   torchvision.utils import save_image



import torch.optim as optim
import sys
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import argparse
import os
import pdb
import math
from   model.Model_Hourglass import *
from   model.GAN import *
from   utils     import calculate_mask,get_mse,get_peak_points,adjust_learning_rate

parser=argparse.ArgumentParser()
parser.add_argument("--img_landmarks", type=str,default="/media/ll/ll/dataset/CelebA/Anno/list_landmarks_align_celeba.txt",  help="path to landmarks of images")
parser.add_argument("--img_root",      type=str,default="/media/ll/ll/dataset/CelebA/Img/img_align_celeba",  help="path to images")
parser.add_argument("--workers",       type=int,default=1,   help="number of data loading workers")
parser.add_argument("--batchSize",     type=int,default=8,  help="input batch size")
parser.add_argument("--nEpochs",       type=int,default=50, help="number of epochs to train for")
parser.add_argument("--cuda",          type=int,default=True,help="number of GPUs to use")
parser.add_argument("--checkpoint",    type=str,default="/home/ll/lwq/Pytorch-face/checkpoint",  help="folder to output model checkpoints")

#####  generator & discriminator
parser.add_argument("--upSample",  type=int, default=6,   help="low to high resolution scaling factor")
parser.add_argument('--imageSize', type=int, default=20,  help='the low resolution image size')
parser.add_argument("--generatorLR",    type=int,default=0.00001,help="learning rate for generator")
parser.add_argument("--discriminatorLR",type=int,default=0.00001,help="learning rate for discriminator")
parser.add_argument("--nResidual",      type=int,default=128,    help="number of Residual block")
#####  FAN- face alignment network
parser.add_argument("--FANLR",type=int,default=0.00001,help="learning rate for face alignment network")



if __name__=="__main__":
	opt=parser.parse_args()
	print (opt)
	if torch.cuda.is_available() and not opt.cuda:
	    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #######   准备数据   ######
	training_dataset=FaceLandmarksDataset(img_dir=opt.img_root,img_txt=opt.img_landmarks)
	training_loader=DataLoader(training_dataset,batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))
	

    #######   模型准备   ######
	generator=Generator(opt.nResidual,opt.upSample)
	discriminator=Discriminator()
	faceAligNet=KFSGNet()
	

	# generator.load_state_dict(torch.load('/home/ll/lwq/Pytorch-face/checkpoint/Generator_model20.pth'))
	# discriminator.load_state_dict(torch.load('/home/ll/lwq/Pytorch-face/checkpoint/Discriminator_model20.pth'))
	# faceAligNet.load_state_dict(torch.load('/home/ll/lwq/Pytorch-face/checkpoint/faceAligNet_model20.pth'))


	generator=generator.cuda()
	discriminator=discriminator.cuda()
	faceAligNet=faceAligNet.cuda()

	######    Loss       ######
	feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).cuda()
	print(feature_extractor)
	adversarial_criterion=nn.BCELoss()
	content_criterion=nn.MSELoss()
	landmarks_critierion=nn.MSELoss()
	
	
	content_criterion=content_criterion.cuda()
	landmarks_critierion=landmarks_critierion.cuda()
	adversarial_criterion=adversarial_criterion.cuda()


	######  optimizor    ######
	optim_generator     = optim.Adam(generator.parameters(), lr=opt.generatorLR)
	optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)
	optim_faceAlignNet  = optim.Adam(faceAligNet.parameters(),lr=opt.FANLR)
	####### epoch start  ######


	ones_const = Variable(torch.ones(opt.batchSize, 1)).cuda()

	print ("SRGAN training")
	for epoch in range(opt.nEpochs):
	# mean_generator_content_loss = 0.0
	# mean_generator_adversarial_loss = 0.0
	# mean_generator_total_loss = 0.0
	# mean_discriminator_loss = 0.0
	# mean_landmarks_criterion_loss=0.0
		
		for i,data in enumerate(training_loader):
			high_res_real,low_res,GT_label=data

			

			#Generator real and fake inputs
			
			high_res_real=Variable(high_res_real.cuda())
			high_res_fake=generator(Variable(low_res.cuda()))
			
			
			high_res_real_label=Variable(GT_label.cuda())
			high_res_real_heatmap=Variable(((torch.from_numpy(OutGaussianMaps(GT_label))).float()).cuda())

			

			target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
			target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()

			#########        Train discriminator        #############
			discriminator.zero_grad()
			# pdb.set_trace()
			discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
			# mean_discriminator_loss += discriminator_loss.data[0]


			#########   PSNR                           #############
			avg_psnr=0
			for k in range(opt.batchSize):
			    criterion_MSE=nn.MSELoss(size_average=True)
			    
			    mse = criterion_MSE(high_res_fake[k], high_res_real[k])
			    psnr = 10 * math.log10(1 / mse.data[0])
			    avg_psnr += psnr

			##########   Train face alignment network   ############
			mask,indices_valid = calculate_mask(high_res_real_heatmap)
			faceAligNet.zero_grad()
			high_res_fake_heatmap=faceAligNet(high_res_fake.detach())
			

			high_res_fake_heatmap = high_res_fake_heatmap * mask
	
			high_res_real_heatmap = high_res_real_heatmap * mask


			# pdb.set_trace()
			faceAligNet_loss=landmarks_critierion(high_res_fake_heatmap,high_res_real_heatmap)

			faceAligNet_loss.backward()
			optim_faceAlignNet.step()			
			discriminator_loss.backward()
			optim_discriminator.step()

			##########       Train generator            ############
			generator.zero_grad()
			real_features = Variable(feature_extractor(high_res_real).data)
			fake_features = feature_extractor(high_res_fake)

			high_res_fake_heatmap = faceAligNet(high_res_fake)
			high_res_fake_heatmap = high_res_fake_heatmap * mask
			high_res_real_heatmap = high_res_real_heatmap * mask
			faceAligNet_loss2=landmarks_critierion(high_res_fake_heatmap,high_res_real_heatmap)

			


			generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)+0.6*faceAligNet_loss2
			generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
			generator_total_loss = generator_content_loss + generator_adversarial_loss

			   # generator_adversarial_loss.backward()



			###########  更新Genertor 参数    ############
			generator_total_loss.backward()
			optim_generator.step()
			##########   adjust learning rate ############
			adjust_learning_rate(optim_generator,epoch)
			adjust_learning_rate(optim_discriminator,epoch)
			adjust_learning_rate(optim_faceAlignNet,epoch)
			########     display              ############
			sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f PSNR:%.4f dB Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch, opt.nEpochs, i, len(training_loader),
			discriminator_loss.data[0],avg_psnr /opt.batchSize, generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))

			if epoch%10==0:
			# 	torch.save(generator.state_dict(),     '%s/generator_final.pth' % opt.checkpoint)
			# 	torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.checkpoint)
				torch.save(generator.state_dict(),    "checkpoint/check3/"+"Generator_model{}.pth".format(epoch))
				torch.save(discriminator.state_dict(),"checkpoint/check3/"+"Discriminator_model{}.pth".format(epoch))
				torch.save(faceAligNet.state_dict(),  "checkpoint/check3/"+"faceAligNet_model{}.pth".format(epoch))
			if i<=10:
				save_image(low_res[0],'/home/ll/lwq/Pytorch-face/output3/'+"epoch"+str(epoch)+"low_res"+str(i)+".jpg")
				save_image(high_res_fake[0].data,'/home/ll/lwq/Pytorch-face/output3/'+"epoch"+str(epoch)+"fake"+str(i)+".jpg")
				save_image(high_res_real[0].data,'/home/ll/lwq/Pytorch-face/output3/'+"epoch"+str(epoch)+"real"+str(i)+".jpg")
			
