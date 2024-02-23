#MNIST VAE retreived from https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

# Modifications:
#Colorize transform that changes the colors of a grayscale image
#colors are chosen from 10 options:
colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]
#specified in "colorvals" variable below
#also there is a skip connection from the first layer to the last layer to enable reconstructions of new stimuli
#and the VAE bottleneck is split, having two different maps
#one is trained with a loss function for color only (eliminating all shape info, reserving only the brightest color)
#the other is trained with a loss function for shape only


# prerequisites
import torch
from dataset_builder import dataset_builder
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import config
from IPython.display import Image, display
import cv2
from PIL import ImageFilter
import imageio, time
import math
import sys
import pandas as pd
from torch.utils.data import DataLoader, Subset

config.init()
#from config import numcolors
global numcolors, colorlabels
from PIL import Image
from mVAE import *
from tokens_capacity import *
import os

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA')
else:
    device = 'cpu'

modelNumber= 1 #which model should be run, this can be 1 through 10

# reload a saved file
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae.eval()
    return vae

#load_checkpoint('output/checkpoint_threeloss_singlegrad200_smfc.pth'.format(modelNumber=modelNumber))
load_checkpoint('output_emnist_recurr/checkpoint_300.pth') # MLR2.0 trained on emnist letters, digits, and fashion mnist

#print('Loading the classifiers')
clf_shapeS=load('classifier_output/ss.joblib')
clf_shapeC=load('classifier_output/sc.joblib')
clf_colorC=load('classifier_output/cc.joblib')
clf_colorS=load('classifier_output/cs.joblib')

#write to a text file
outputFile = open('outputFile.txt'.format(modelNumber),'w')

bs_testing = 1000     # number of images for testing. 20000 is the limit
shape_coeff = 1       #cofficient of the shape map
color_coeff = 1       #coefficient of the color map
location_coeff = 0    #Coefficient of Location map
l1_coeff = 1          #coefficient of layer 1
l2_coeff = 1          #coefficient of layer 2
shapeLabel_coeff= 1   #coefficient of the shape label
colorLabel_coeff = 1  #coefficient of the color label
location_coeff = 0  #coefficient of the color label

bpsize = 2500         #size of the binding pool
token_overlap =.4
bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item

normalize_fact_familiar=1
normalize_fact_novel=1


imgsize = 28
all_imgs = []

#number of repetions for statistical inference
hugepermnum=10000
bigpermnum = 500
smallpermnum = 100

Fig2aFlag = 0       #binding pool reconstructions   NOTWORKING
fig_new_loc = 0     # reconstruct retina images with digits in the location opposite of training
fig_loc_compare = 1 # compare retina images with digits in the same location as training and opposite location  
Fig2bFlag = 1        #novel objects stored and retrieved from memory, one at a time
Fig2btFlag = 1        #novel objects stored and retrieved from memory, in tokens
Fig2cFlag = 1       #familiar objects stored and retrieved from memory, using tokens 
sampleflag = 0   #generate random objects from latents (plot working, not behaving as expected)

    
bindingtestFlag = 0  #simulating binding shape-color of two items  NOT WORKING

Tab1Flag_noencoding = 0 #classify reconstructions (no memory) NOT WORKINGy
Tab1Flag = 0 #             #classify binding pool memoriesNOT WORKING
Tab1SuppFlag = 0        #memory of labels (this is table 1 + Figure 2 in supplemental which includes the data in Figure 3)
Tab2Flag =0 #NOT WORKING
TabTwoColorFlag = 0  #NOT WORKING
TabTwoColorFlag1 = 0            #Cross correlations for familiar vs novel  #NOT WORKING
noveltyDetectionFlag=0  #detecting whether a stimulus is familiar or not  #NOT WORKING
latents_crossFlag = 0   #Cross correlations for familiar vs novel for when infromation is stored from the shape/color maps vs. L1. versus straight reconstructions
                        #This Figure is not included in the paper  #NOT WORKING

bs=100   # number of samples to extract from the dataset

#### generate some random samples (currently commented out due to cuda errors)  #NOT WORKING
if (sampleflag):
    zc=torch.randn(64,16).cuda()*1
    zs=torch.randn(64,16).cuda()*1
    with torch.no_grad():
        sample = vae.decoder_cropped(zs,zc,0).cuda()
        sample_c= vae.decoder_cropped(zs*0,zc,0).cuda()
        sample_s = vae.decoder_cropped(zs, zc*0, 0).cuda()
        sample=sample.view(64, 3, 28, 28)
        sample_c=sample_c.view(64, 3, 28, 28)
        sample_s=sample_s.view(64, 3, 28, 28)
        save_image(sample[0:8], 'output{num}/sample.png'.format(num=modelNumber))
        save_image(sample_c[0:8], 'output{num}/sample_color.png'.format(num=modelNumber))
        save_image(sample_s[0:8], 'output{num}/sample_shape.png'.format(num=modelNumber))

    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))
    test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True, num_workers=nw)


######################## Figure 2a #######################################################################################
#store items using both features, and separately color and shape (memory retrievals)


if Fig2aFlag==1:
    print('generating figure 2a, reconstructions from the binding pool')

    numimg= 6
    bs=numimg #number of images to display in this figure
    nw=2
    bs_testing = numimg # 20000 is the limit
    train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            None,False) 

    test_loader_smaller = test_loader_noSkip
    images, shapelabels = next(iter(test_loader_smaller))#peel off a large number of images
    #orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
    imgs = images.clone().cuda()

    #run them all through the encoder
    l1_act, l2_act, shape_act, color_act, location_act = activations(imgs)  #get activations from this small set of images

    #binding pool outputs
    BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk =            BP(bpPortion , l1_act, l2_act, shape_act, color_act, location_act, shape_coeff, color_coeff, location_coeff, l1_coeff,l2_coeff,normalize_fact_familiar)
    BP_in, shape_out_BP_shapeonly,  color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion , l1_act, l2_act, shape_act, color_act, location_act, shape_coeff, 0,0,0,0,normalize_fact_familiar)
    BP_in,  shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion , l1_act, l2_act, shape_act, color_act, 0, color_coeff,0,0,normalize_fact_familiar)
    BP_in,  shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion , l1_act, l2_act, shape_act, color_act, 0, 0,l1_coeff,0,normalize_fact_familiar)
    BP_in,  shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion , l1_act, l2_act, shape_act, color_act, 0, 0,0,l2_coeff,normalize_fact_familiar)

    #memory retrievals from Bottleneck storage
    bothRet = vae.decoder_noskip(shape_out_BP_both, color_out_BP_both, 0).cuda()  # memory retrieval from the bottleneck
    shapeRet = vae.decoder_shape(shape_out_BP_shapeonly, color_out_BP_shapeonly , 0).cuda()  #memory retrieval from the shape map
    colorRet = vae.decoder_color(shape_out_BP_coloronly, color_out_BP_coloronly, 0).cuda()  #memory retrieval from the color map

    save_image(
        torch.cat([imgs[0: numimg].view(numimg, 3, 28, 28), bothRet[0: numimg].view(numimg, 3, 28, 28),
                   shapeRet[0: numimg].view(numimg, 3, 28, 28), colorRet[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_BP_bottleneck_.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )

    #memory retrievals when information was stored from L1 and L2
    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 1, 'noskip') #bp retrievals from layer 1
    BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 2, 'noskip') #bp retrievals from layer 2

    save_image(
        torch.cat([
                   BP_layer2_noskip[0: numimg].view(numimg, 3, 28, 28), BP_layer1_noskip[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_layer2_layer1.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )


if fig_new_loc == 1:
    #recreate images of digits, but on the opposite side of the retina that they had originally been trained on 
    #no working memory, just a reconstruction
    bs = 100
    retina_size = 100  #how wide is the retina

    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            {},True,{'left':list(range(0,5)),'right':list(range(5,10))}) 
    
    #Code showing the data loader for how the model was trained, empty dict in 3rd param is for any color:
    '''train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            {},True,{'right':list(range(0,5)),'left':list(range(5,10))}) '''    

    dataiter_noSkip = iter(test_loader_noSkip)
    data = dataiter_noSkip.next()
    data = data[0] #.cuda()
    
    sample_data = data
    sample_size = 15
    sample_data[0] = sample_data[0][:sample_size]
    sample_data[1] = sample_data[1][:sample_size]
    sample_data[2] = sample_data[2][:sample_size]
    sample = sample_data
    with torch.no_grad():  #generate reconstructions for these stimuli from different pathways through the model
        reconl, mu_color, log_var_color, mu_shape, log_var_shape,mu_location, log_var_location = vae(sample, 'location') #location
        reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'retinal') #retina
        recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
        reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
        recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape
            
    empty_retina = torch.zeros((sample_size, 3, 28, 100))

    #repackage the reconstructions for visualization
    n_reconl = empty_retina.clone()
    

    for i in range(len(reconl)):
        n_reconl[i][0, :, 0:100] = reconl[i]
        n_reconl[i][1, :, 0:100] = reconl[i]
        n_reconl[i][2, :, 0:100] = reconl[i]


    n_recond = empty_retina.clone()
    for i in range(len(recond)):
        n_recond[i][0, :, 0:imgsize] = recond[i][0]
        n_recond[i][1, :, 0:imgsize] = recond[i][1]
        n_recond[i][2, :, 0:imgsize] = recond[i][2]

    n_reconc = empty_retina.clone()
    for i in range(len(reconc)):
        n_reconc[i][0, :, 0:28] = reconc[i][0]
        n_reconc[i][1, :, 0:28] = reconc[i][1]
        n_reconc[i][2, :, 0:28] = reconc[i][2]

    n_recons = empty_retina.clone()
    for i in range(len(recons)):
        n_recons[i][0, :, 0:28] = recons[i][0]
        n_recons[i][1, :, 0:28] = recons[i][1]
        n_recons[i][2, :, 0:28] = recons[i][2]
    line1 = torch.ones((1,2)) * 0.5
    line1 = line1.view(1,1,1,2)
    line1 = line1.expand(sample_size, 3, imgsize, 2)
    
    n_reconc = torch.cat((n_reconc,line1),dim = 3).cuda()
    n_recons = torch.cat((n_recons,line1),dim = 3).cuda()
    n_reconl = torch.cat((n_reconl,line1),dim = 3).cuda()
    n_recond = torch.cat((n_recond,line1),dim = 3).cuda()
    shape_color_dim = retina_size + 2
    sample = torch.cat((sample[0],line1),dim = 3).cuda()
    
    reconb = torch.cat((reconb,line1.cuda()),dim = 3).cuda()
    utils.save_image(
        torch.cat([sample.view(sample_size, 3, imgsize, retina_size+2), reconb.view(sample_size, 3, imgsize, retina_size+2), n_recond.view(sample_size, 3, imgsize, retina_size+2),
                    n_reconl.view(sample_size, 3, imgsize, retina_size+2), n_reconc.view(sample_size, 3, imgsize, shape_color_dim), n_recons.view(sample_size, 3, imgsize, shape_color_dim)], 0),
                'output{num}/figure_new_location.png'.format(num=modelNumber),
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

if fig_loc_compare == 1:
    bs = 15
    train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            None,True,{'left':list(range(0,5)),'right':list(range(5,10))})    
    imgsize = 28
    numimg = 10
    
    print(type(test_loader_noSkip))
    
    dataiter_noSkip_test = iter(test_loader_noSkip)
    dataiter_noSkip_train = iter(train_loader_noSkip)
    skipd = iter(train_loader_skip)
    skip = skipd.next()
    print(skip[0].size())
    print(type(dataiter_noSkip_test))
    data_test = dataiter_noSkip_test.next()
    data_train = dataiter_noSkip_train.next()

    data = data_train[0].copy()
    #print(data.size())
    data[0] = torch.cat((data_test[0][0], data_train[0][0]),dim=0) #.cuda()
    data[1] = torch.cat((data_test[0][1], data_train[0][1]),dim=0)
    data[2] = torch.cat((data_test[0][2], data_train[0][2]),dim=0)

    sample = data
    sample_size = 15
    print(sample[0].size(),sample[1].size(),sample[2].size())
    with torch.no_grad():
        reconl, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'location') #location
        reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'retinal') #retina
        recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
        reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
        recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape
            
    empty_retina = torch.zeros((2*sample_size, 3, 28, 100))

    n_reconl = empty_retina.clone()
    for i in range(len(reconl)):
        n_reconl[i][0, :, 0:100] = reconl[i]
        n_reconl[i][1, :, 0:100] = reconl[i]
        n_reconl[i][2, :, 0:100] = reconl[i]

    n_recond = empty_retina.clone()
    for i in range(len(recond)):
        n_recond[i][0, :, 0:imgsize] = recond[i][0]
        n_recond[i][1, :, 0:imgsize] = recond[i][1]
        n_recond[i][2, :, 0:imgsize] = recond[i][2]

    n_reconc = empty_retina.clone()
    for i in range(len(reconc)):
        n_reconc[i][0, :, 0:28] = reconc[i][0]
        n_reconc[i][1, :, 0:28] = reconc[i][1]
        n_reconc[i][2, :, 0:28] = reconc[i][2]

    n_recons = empty_retina.clone()
    for i in range(len(recons)):
        n_recons[i][0, :, 0:28] = recons[i][0]
        n_recons[i][1, :, 0:28] = recons[i][1]
        n_recons[i][2, :, 0:28] = recons[i][2]
    line1 = torch.ones((1,2)) * 0.5
    line1 = line1.view(1,1,1,2)
    line2 = line1.expand(sample_size, 3, imgsize, 2)
    line1 = line1.expand(2*sample_size, 3, imgsize, 2)
    
    n_reconc = torch.cat((n_reconc,line1),dim = 3).cuda()
    n_recons = torch.cat((n_recons,line1),dim = 3).cuda()
    n_reconl = torch.cat((n_reconl,line1),dim = 3).cuda()
    n_recond = torch.cat((n_recond,line1),dim = 3).cuda()
    shape_color_dim = retina_size + 2
    sample_test = torch.cat((sample[0][:sample_size],line2),dim = 3).cuda()
    sample_train = torch.cat((sample[0][sample_size:(2*sample_size)],line2),dim = 3).cuda()
    reconb = torch.cat((reconb,line1.cuda()),dim = 3).cuda()
    utils.save_image(
        torch.cat((
        torch.cat([sample_train.view(sample_size, 3, imgsize, retina_size+2), reconb[sample_size:(2*sample_size)].view(sample_size, 3, imgsize, retina_size+2), n_reconl[sample_size:(2*sample_size)].view(sample_size, 3, imgsize, retina_size+2),
                    n_recond[sample_size:(2*sample_size)].view(sample_size, 3, imgsize, retina_size+2), n_reconc[sample_size:(2*sample_size)].view(sample_size, 3, imgsize, shape_color_dim), n_recons[sample_size:(2*sample_size)].view(sample_size, 3, imgsize, shape_color_dim)], 0),
        torch.cat([sample_test.view(sample_size, 3, imgsize, retina_size+2), reconb[:(sample_size)].view(sample_size, 3, imgsize, retina_size+2), n_reconl[:(sample_size)].view(sample_size, 3, imgsize, retina_size+2),
                    n_recond[:(sample_size)].view(sample_size, 3, imgsize, retina_size+2), n_reconc[:(sample_size)].view(sample_size, 3, imgsize, shape_color_dim), n_recons[:(sample_size)].view(sample_size, 3, imgsize, shape_color_dim)], 0)),0),
                'output{num}/figure_new_location.png'.format(num=modelNumber),
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )
    image_pil = Image.open('output{num}/figure_new_location.png'.format(num=modelNumber))
    trained_label = "Trained Data"
    untrained_label = "Untrained Data"
    # Add trained and untrained labels to the image using PIL's Draw module
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()  # You can choose a different font or size

    # Trained data label at top left
    trained_label_position = (10, 10)  # Adjust the position of the text
    draw.text(trained_label_position, trained_label, fill=(255, 255, 255), font=font)

    # Untrained data label at bottom left
    image_width, image_height = image_pil.size
    untrained_label_position = (10, image_height//2)  # Adjust the position of the text
    draw.text(untrained_label_position, untrained_label, fill=(255, 255, 255), font=font)

    # Save the modified image with labels
    image_pil.save('output{num}/figure_new_location.png'.format(num=modelNumber))

    print("Images with labels saved successfully.")

if Fig2bFlag==1:
    all_imgs = []
    print('generating Figure 2b, Novel characters retrieved from memory of L1 and Bottleneck')
    retina_size = 100
    imgsize = 28
    numimg = 7

    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        img_new = convert_tensor(Image.open(f'current_bengali/{i}_thick.png'))[0:3,:,:]
        #img_new = Colorize_func(img)   # Currently broken, but would add a color to each
        all_imgs.append(img_new)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()
    location = torch.zeros(imgs.size()[0], vae.l_dim).cuda()
    location[0] = 1
    #push the images through the encoder
    l1_act, l2_act, shape_act, color_act, location_act = activations(imgs.view(-1,3,28,28), location)
    
    imgmatrixL1skip  = torch.empty((0,3,28,28)).cuda()
    imgmatrixL1noskip  = torch.empty((0,3,28,28)).cuda()
    imgmatrixMap  = torch.empty((0,3,28,28)).cuda()
    
    #now run them through the binding pool!
    #store the items and then retrive them, and do it separately for shape+color maps, then L1, then L2. 
    #first store and retrieve the shape, color and location maps
    
    for n in range (0,numimg):
            # reconstruct directly from activation
        recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act.view(numimg,-1), l2_act, 3, 'skip_cropped')
        
        #now store/retrieve from L1
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act[n,:].view(1,-1), l2_act[n,:].view(1,-1), shape_act[n,:].view(1,-1),color_act[n,:].view(1,-1),location_act[n,:].view(1,-1),0, 0,0,1,0,1,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, BP_layer2_out, BP_layerI_out = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1,normalize_fact_novel)
      
        # reconstruct  from BP version of layer 1, run through the skip
        BP_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out.view(1,-1),BP_layer2_out,3, 'skip_cropped')

        # reconstruct  from BP version of layer 1, run through the bottleneck
        BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out.view(1,-1),BP_layer2_out, 3, 'cropped')
        
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act[n,:].view(1,-1), l2_act[n,:].view(1,-1), shape_act[n,:].view(1,-1),color_act[n,:].view(1,-1),location_act[n,:].view(1,-1),1, 1,0,0,0,1,normalize_fact_novel)
        shape_out_BP, color_out_BP, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1,normalize_fact_novel)
      
        #reconstruct from BP version of the shape and color maps
        retrievals = vae.decoder_cropped(shape_out_BP, color_out_BP,0,0).cuda()

        imgmatrixL1skip = torch.cat([imgmatrixL1skip,BP_layer1_skip])
        imgmatrixL1noskip = torch.cat([imgmatrixL1noskip,BP_layer1_noskip])
        imgmatrixMap= torch.cat([imgmatrixMap,retrievals])

    #save an image showing:  original images, reconstructions directly from L1,  from L1 BP, from L1 BP through bottleneck, from maps BP
    save_image(torch.cat([imgs[0: numimg].view(numimg, 3, 28, imgsize), imgmatrixL1skip, imgmatrixL1noskip, imgmatrixMap], 0),'output{num}/figure2b.png'.format(num=modelNumber),
            nrow=numimg,            normalize=False, range=(-1, 1),)  

if Fig2btFlag==1:
    all_imgs = []
    recon = list()
    print('generating Figure 2bt, Novel characters retrieved from memory of L1 and Bottleneck using Tokens')
    retina_size = 100
    imgsize = 28
    numimg = 7  #how many objects will we use here?

    #load in some examples of Bengali Characters
    for i in range (1,numimg+1):
        img_new = convert_tensor(Image.open(f'current_bengali/{i}_thick.png'))[0:3,:,:]
        all_imgs.append(img_new)

    #all_imgs is a list of length 3, each of which is a 3x28x28
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * imgsize * imgsize).cuda()   #dimensions are R+G+B + # pixels

    imgmatrix = imgs.view(numimg,3,28,28)
    #push the images through the model
    l1_act, l2_act, shape_act, color_act, location_act = activations(imgs.view(-1,3,28,28))
    emptyshape = torch.empty((1,3,28,28)).cuda()
    # store 1 -> numimg items
    for n in range(1,numimg+1):
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,0, 0,0,1,0,n,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,n,normalize_fact_novel)
      
        recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_out_all.view(n,-1), l2_act, 3, 'skip_cropped')        
        imgmatrix= torch.cat([imgmatrix,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrix= torch.cat([imgmatrix,emptyshape*0],0)

    save_image(imgmatrix,'output{num}/figure2bt.png'.format(num=modelNumber),    nrow=numimg, normalize=False,  range=(-1, 1),   )



if Fig2cFlag==1:

    print('generating Figure 2c, Familiar characters retrieved from Bottleneck using Tokens')
    retina_size = 100
    reconMap = list()
    reconL1 = list()
    imgsize = 28
    numimg = 7  #how many objects will we use here?

    #make the data loader, but specifically we are creating stimuli on the opposite to how the model was trained
    train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            {},True,{'left':list(range(0,5)),'right':list(range(5,10))}) 
    
    #Code showing the data loader for how the model was trained, empty dict in 3rd param is for any color:
    '''train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder('mnist',bs,
            {},True,{'right':list(range(0,5)),'left':list(range(5,10))}) '''    

    dataiter_noSkip = iter(test_loader_noSkip)
    data = dataiter_noSkip.next()
    data = data[0] #.cuda()
    
    sample_data = data
    sample_size = numimg
    sample_data[0] = sample_data[0][:sample_size]
    sample_data[1] = sample_data[1][:sample_size]
    sample = sample_data
    
    
    #push the images through the model
    l1_act, l2_act, shape_act, color_act, location_act = activations(sample[1].view(-1,3,28,28).cuda())
    emptyshape = torch.empty((1,3,28,28)).cuda()
    imgmatrixMap = sample[1].view(numimg,3,28,28).cuda()
    imgmatrixL1 = sample[1].view(numimg,3,28,28).cuda()

    # store 1 -> numimg items
    for n in range(1,numimg+1):
        #Store and retrieve the map versions
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,1, 1,0,0,0,n,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,n,normalize_fact_novel)
        retrievals = vae.decoder_cropped(shape_out_all, color_out_all,0,0).cuda()

        #Store and retrieve the L1 version
        BPOut, Tokenbindings = BPTokens_storage(bpsize, bpPortion, l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,0, 0,0,1,0,n,normalize_fact_novel)
        shape_out_all, color_out_all, location_out_all, l2_out_all, l1_out_all = BPTokens_retrieveByToken( bpsize, bpPortion, BPOut, Tokenbindings,l1_act.view(numimg,-1), l2_act.view(numimg,-1), shape_act,color_act,location_act,n,normalize_fact_novel)
        recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_out_all.view(n,-1), l2_act, 3, 'skip_cropped')        
        
        imgmatrixMap= torch.cat([imgmatrixMap,retrievals],0)
        imgmatrixL1= torch.cat([imgmatrixL1,recon_layer1_skip],0)

        #now pad with empty images
        for i in range(n,numimg):
            imgmatrixMap= torch.cat([imgmatrixMap,emptyshape*0],0)
            imgmatrixL1= torch.cat([imgmatrixL1,emptyshape*0],0)
 
    save_image(imgmatrixL1, 'output{num}/figure2cL1.png'.format(num=modelNumber),  nrow=numimg,        normalize=False,range=(-1, 1))
    save_image(imgmatrixMap, 'output{num}/figure2cMap.png'.format(num=modelNumber),  nrow=numimg,        normalize=False,range=(-1, 1))
        

###################Table 2##################################################
if Tab2Flag ==1:

    numModels=10

    print('Tab2 loss of quality of familiar vs novel items using correlation')

    setSizes=[1,2,3,4] #number of tokens


    familiar_corr_all=list()
    familiar_corr_all_se=list()
    novel_corr_all=list()
    novel_corr_all_se=list()

    familiar_skip_all=list()
    familiar_skip_all_se=list()

    novel_BN_all=list()
    novel_BN_all_se = list()

    perms = bigpermnum#number of times it repeats storing/retrieval



    for numItems in setSizes:

        familiar_corr_models = list()
        novel_corr_models = list()
        familiar_skip_models=list()
        novel_BN_models=list()

        print('SetSize {num}'.format(num=numItems))

        for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10

            load_checkpoint(
                'output{modelNumber}/checkpoint_threeloss_singlegrad50.pth'.format(modelNumber=modelNumber))

            # reset the data set for each set size
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=nw)

            # This function is in tokens_capacity.py

            familiar_corrValues= storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0,1)

            familiar_corrValues_skip = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                      test_loader_smaller, 'fam', 1, 1)


            novel_corrValues = storeretrieve_crosscorrelation_test(numItems, perms, bpsize,
                                                                                             bpPortion, shape_coeff,
                                                                                             color_coeff,
                                                                                             normalize_fact_familiar,
                                                                                             normalize_fact_novel,
                                                                                             modelNumber,
                                                                                             test_loader_smaller, 'nov',
                                                                                             1,1)
            novel_corrValues_BN = storeretrieve_crosscorrelation_test(numItems, perms, bpsize,
                                                                   bpPortion, shape_coeff,
                                                                   color_coeff,
                                                                   normalize_fact_familiar,
                                                                   normalize_fact_novel,
                                                                   modelNumber,
                                                                   test_loader_smaller, 'nov',
                                                                   0, 1)


            familiar_corr_models.append(familiar_corrValues)
            familiar_skip_models.append(familiar_corrValues_skip)
            novel_corr_models.append(novel_corrValues)
            novel_BN_models.append(novel_corrValues_BN)




        familiar_corr_models_all=np.array(familiar_corr_models).reshape(-1,1)
        novel_corr_models_all = np.array(novel_corr_models).reshape(1, -1)

        familiar_skip_models_all=np.array(familiar_skip_models).reshape(1,-1)
        novel_BN_models_all=np.array(novel_BN_models).reshape(1,-1)





        familiar_corr_all.append(np.mean(familiar_corr_models_all))
        familiar_corr_all_se.append(np.std(familiar_corr_models_all)/math.sqrt(numModels))


        novel_corr_all.append(np.mean( novel_corr_models_all))
        novel_corr_all_se.append(np.std(novel_corr_models_all)/math.sqrt(numModels))

        familiar_skip_all.append(np.mean(familiar_skip_models_all))
        familiar_skip_all_se.append(np.std(familiar_skip_models_all)/math.sqrt(numModels))

        novel_BN_all.append(np.mean(novel_BN_models_all))
        novel_BN_all_se.append(np.std(novel_BN_models_all)/math.sqrt(numModels))

    #the mean correlation value between input and recontructed images for familiar and novel stimuli
    outputFile.write('Familiar correlation\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE  {2:.3g}\n'.format(setSizes[i],familiar_corr_all[i],familiar_corr_all_se[i]))



    outputFile.write('\nfNovel correlation\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], novel_corr_all[i], novel_corr_all_se[i]))

    outputFile.write('\nfamiliar correlation vis skip \n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], familiar_skip_all[i], familiar_skip_all_se[i]))

    outputFile.write('\nnovel correlation via BN \n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], novel_BN_all[i], novel_BN_all_se[i]))

    #This part (not included in the paper) visualizes the cross correlation between novel shapes retrieved from the skip and familiar shapes retrived from the BN
    plt.figure()
    familiar_corr_all=np.array(familiar_corr_all)

    novel_corr_all=np.array(novel_corr_all)

    plt.errorbar(setSizes,familiar_corr_all,yerr=familiar_corr_all_se, fmt='o',markersize=3)
    plt.errorbar(setSizes, novel_corr_all, yerr=novel_corr_all_se, fmt='o', markersize=3)


    plt.axis([0,6, 0, 1])
    plt.xticks(np.arange(0,6,1))
    plt.show()

#############################################
if latents_crossFlag ==1:

    numModels=10

    print('cross correlations for familiar items when reconstructed and when retrived from BN or L1+skip ')

    setSizes=[1,2,3,4] #number of tokens

    noskip_recon_mean=list()
    noskip_recon_se=list()
    noskip_ret_mean=list()
    noskip_ret_se=list()

    skip_recon_mean=list()
    skip_recon_se=list()

    skip_ret_mean=list()
    skip_ret_se=list()

    perms = bigpermnum #number of times it repeats storing/retrieval

    for numItems in setSizes:

        noskip_Reconmodels=list()
        noskip_Retmodels=list()

        skip_Reconmodels=list()
        skip_Retmodels=list()

        print('SetSize {num}'.format(num=numItems))

        for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10

            load_checkpoint(
                'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

            # reset the data set for each set size
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=nw)

            # This function is in tokens_capacity.py
            #familiar items reconstrcuted via BN with no memory
            noskip_noMem= storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0, 0)
            # familiar items retrieved via BN
            noskip_Mem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0, 1)
            #recon from L1
            skip_noMem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                     test_loader_smaller, 'fam', 1, 0)
             #retrieve from L1 +skip
            skip_Mem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                      test_loader_smaller, 'fam', 1, 1)


            noskip_Reconmodels.append(noskip_noMem)
            noskip_Retmodels.append(noskip_Mem)

            skip_Reconmodels.append(skip_noMem)
            skip_Retmodels.append(skip_Mem)




        noskip_Reconmodels_all=np.array(noskip_Reconmodels).reshape(-1,1)
        noskip_Retmodels_all=np.array(noskip_Retmodels).reshape(-1,1)
        skip_Reconmodels_all = np.array(skip_Reconmodels).reshape(1, -1)
        skip_Retmodels_all=np.array(skip_Retmodels).reshape(1,-1)




        noskip_recon_mean.append(np.mean(noskip_Reconmodels_all))
        noskip_recon_se.append(np.std(noskip_Reconmodels_all)/math.sqrt(numModels))

        noskip_ret_mean.append(np.mean(noskip_Retmodels_all))
        noskip_ret_se.append(np.std(noskip_Retmodels_all) / math.sqrt(numModels))

        skip_recon_mean.append(np.mean(skip_Reconmodels_all))
        skip_recon_se.append(np.std(skip_Reconmodels_all) / math.sqrt(numModels))

        skip_ret_mean.append(np.mean(skip_Retmodels_all))
        skip_ret_se.append(np.std(skip_Retmodels_all) / math.sqrt(numModels))






    #the mean correlation value between input and recontructed images for familiar and novel stimuli
    outputFile.write('correlation for recons from BN\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE  {2:.3g}\n'.format(setSizes[i],noskip_recon_mean[i],noskip_recon_se[i]))

    outputFile.write('\nCorrelation for retrievals from BN\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i],noskip_ret_mean[i],noskip_ret_se[i]))

    outputFile.write('\ncorrelation for recons from skip\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], skip_recon_mean[i], skip_recon_se[i]))

    outputFile.write('\ncorrelation for retrievals from skip\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], skip_ret_mean[i], skip_ret_se[i]))

    plt.figure()

    correlations=np.array([skip_recon_mean,noskip_recon_mean, skip_ret_mean,noskip_ret_mean]).squeeze()

    corr_se=np.array([skip_recon_se,noskip_recon_se, skip_ret_se,noskip_ret_se]).squeeze()


    fig, ax = plt.subplots()
    pos=np.array([1,2,3,4])

    ax.bar(pos, correlations, yerr=corr_se, width=.4, alpha=.6, ecolor='black', color=['blue', 'blue', 'red', 'red'])


    plt.show()



########################  Ability to extract the correct token from a shape-only stimulus

if bindingtestFlag ==1:
    numModels=10
    perms = bigpermnum
    correctToken=np.tile(0.0,numModels)
    correctToken_diff=np.tile(0.0,numModels)
    accuracyColor=np.tile(0.0,numModels)
    accuracyColor_diff=np.tile(0.0,numModels)
    accuracyShape=np.tile(0.0,numModels)
    accuracyShape_diff=np.tile(0.0,numModels)

    for modelNumber in range(1,numModels+1):


        print('testing binding cue retrieval')



        # grey shape cue binding accuracy for only two items when the items are the same (e.g. two 3's ).


        bs_testing = 2
        correctToken[modelNumber-1],accuracyColor[modelNumber-1],accuracyShape[modelNumber-1] = binding_cue(bs_testing, perms, bpsize, bpPortion, shape_coeff, color_coeff, 'same',
                                                modelNumber)


        # grey shape cue binding accuracy for only two items when the two items are different
        correctToken_diff[modelNumber-1],accuracyColor_diff[modelNumber-1] ,accuracyShape_diff[modelNumber-1] = binding_cue(bs_testing, perms, bpsize, bpPortion, shape_coeff, color_coeff
                                                         , 'diff', modelNumber)


    correctToekn_all= correctToken.mean()
    SD=correctToken.std()

    correctToekn_diff_all=correctToken_diff.mean()
    SD_diff=correctToken_diff.std()
    accuracyColor_all=accuracyColor.mean()
    SD_color= accuracyColor.std()
    accuracyColor_diff_all=accuracyColor_diff.mean()
    SD_color_diff=accuracyColor_diff.std()

    accuracyShape_all=accuracyShape.mean()
    SD_shape= accuracyShape.std()
    accuracyShape_diff_all=accuracyShape_diff.mean()
    SD_shape_diff=accuracyShape_diff.std()



    outputFile.write('the correct retrieved token for same shapes condition is: {num} and SD is {sd}'.format(num=correctToekn_all, sd=SD))
    outputFile.write('\n the correct retrieved color for same shapes condition is: {num} and SD is {sd}'.format(num=accuracyColor_all, sd=SD_color))
    outputFile.write('\n the correct retrieved shape for same shapes condition is: {num} and SD is {sd}'.format(num=accuracyShape_all, sd=SD_shape))

    outputFile.write(
        '\n the correct retrieved token for different shapes condition is: {num} and SD is {sd}'.format(num=correctToekn_diff_all, sd=SD_diff))
    outputFile.write(
        '\n the correct retrieved color for different shapes condition is: {num} and SD is {sd}'.format(num=accuracyColor_diff_all, sd=SD_color_diff))
    outputFile.write(
        '\n the correct retrieved shape for different shapes condition is: {num} and SD is {sd}'.format(num=accuracyShape_diff_all, sd=SD_shape_diff))










#############Table 1 for the no memmory condition#####################
numModels = 1

perms=100

if Tab1Flag_noencoding == 1:

    print('Table 1 shape labels predicted by the classifier before encoded in memory')


    SSreport = np.tile(0.0,[perms,numModels])
    SCreport = np.tile(0.0,[perms,numModels])
    CCreport = np.tile(0.0,[perms,numModels])
    CSreport = np.tile(0.0,[perms,numModels])



    for temp in range(1,numModels +1):  # which model should be run, this can be 1 through 10

        modelNumber = 5
        load_checkpoint('output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

        print('doing model {0} for Table 1'.format(modelNumber))
        clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
        clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
        clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
        clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))

        for rep in range(0,perms):

           pred_cc, pred_cs, CCreport[rep,modelNumber - 1], CSreport[rep,modelNumber - 1] = classifier_color_test('noskip',
                                                                                                       clf_colorC,
                                                                                                       clf_colorS)

           pred_ss, pred_sc, SSreport[rep,modelNumber-1], SCreport[rep,modelNumber-1] = classifier_shape_test('noskip', clf_shapeS, clf_shapeC)


    print(CCreport)
    CCreport=CCreport.reshape(1,-1)
    CSreport=CSreport.reshape(1,-1)
    SSreport=SSreport.reshape(1,-1)
    SCreport=SCreport.reshape(1,-1)


    outputFile.write('Table 1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g}\n'.format(SSreport.mean(),SSreport.std()/math.sqrt(numModels*perms), SCreport.mean(),  SCreport.std()/math.sqrt(numModels) ))
    outputFile.write('Table 1, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g}\n'.format(CCreport.mean(),CCreport.std()/math.sqrt(numModels*perms), CSreport.mean(),  CSreport.std()/math.sqrt(numModels)))


########################## Table 1 for memory conditions ######################################################################

if Tab1Flag == 1:
    numModels=1
    perms=1
    SSreport_both = np.tile(0.0, [perms,numModels])
    SCreport_both = np.tile(0.0, [perms,numModels])
    CCreport_both = np.tile(0.0, [perms,numModels])
    CSreport_both = np.tile(0.0, [perms,numModels])
    SSreport_shape = np.tile(0.0, [perms,numModels])
    SCreport_shape = np.tile(0.0, [perms,numModels])
    CCreport_shape = np.tile(0.0, [perms,numModels])
    CSreport_shape = np.tile(0.0, [perms,numModels])
    SSreport_color = np.tile(0.0, [perms,numModels])
    SCreport_color = np.tile(0.0, [perms,numModels])
    CCreport_color = np.tile(0.0, [perms,numModels])
    CSreport_color = np.tile(0.0, [perms,numModels])
    SSreport_l1 = np.tile(0.0, [perms,numModels])
    SCreport_l1= np.tile(0.0, [perms,numModels])
    CCreport_l1 = np.tile(0.0, [perms,numModels])
    CSreport_l1 = np.tile(0.0, [perms,numModels])
    SSreport_l2 = np.tile(0.0, [perms,numModels])
    SCreport_l2 = np.tile(0.0, [perms,numModels])
    CCreport_l2 = np.tile(0.0, [perms,numModels])
    CSreport_l2 = np.tile(0.0, [perms,numModels])


    for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
        load_checkpoint(
            'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

        print('doing model {0} for Table 1'.format(modelNumber))
        clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
        clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
        clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
        clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


        print('Doing Table 1')

        for rep in range(0,perms):
            numcolors = 0

            colorlabels = thecolorlabels(test_dataset)

            bs_testing = 1000  # 20000 is the limit
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True,
                                                          num_workers=nw)
            colorlabels = colorlabels[0:bs_testing]

            images, shapelabels = next(iter(test_loader_smaller))  # peel off a large number of images
            orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
            imgs = orig_imgs.clone()

            # run them all through the encoder
            l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

            # now store and retrieve them from the BP
            BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                         shape_act, color_act,
                                                                                         shape_coeff, color_coeff, 1, 1,
                                                                                         normalize_fact_familiar)
            BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                                   l2_act, shape_act,
                                                                                                   color_act,
                                                                                                   shape_coeff, 0, 0, 0,
                                                                                                   normalize_fact_familiar)
            BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                                   l2_act, shape_act,
                                                                                                   color_act, 0,
                                                                                                   color_coeff, 0, 0,
                                                                                                   normalize_fact_familiar)

            BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                        shape_act, color_act, 0, 0,
                                                                                        l1_coeff, 0,
                                                                                        normalize_fact_familiar)

            BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                        shape_act, color_act, 0, 0, 0,
                                                                                        l2_coeff,
                                                                                        normalize_fact_familiar)

           # Table 1: classifier accuracy for shape and color for memory retrievals
            print('classifiers accuracy for memory retrievals of BN_both for Table 1')
            pred_ss, pred_sc, SSreport_both[rep,modelNumber-1], SCreport_both[rep,modelNumber-1] = classifier_shapemap_test_imgs(shape_out_BP_both, shapelabels,
                                                                             colorlabels, bs_testing, clf_shapeS,
                                                                             clf_shapeC)
            pred_cc, pred_cs, CCreport_both[rep,modelNumber-1], CSreport_both[rep,modelNumber-1]= classifier_colormap_test_imgs(color_out_BP_both, shapelabels,
                                                                             colorlabels, bs_testing, clf_colorC,clf_colorS)



            # Table 1: classifier accuracy for shape and color for memory retrievals
            print('classifiers accuracy for memory retrievals of BN_shapeonly for Table 1')
            pred_ss, pred_sc, SSreport_shape[rep,modelNumber - 1], SCreport_shape[
            rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_shapeonly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_shape[rep,modelNumber - 1], CSreport_shape[
            rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_shapeonly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_colorC, clf_colorS)

            print('classifiers accuracy for memory retrievals of BN_coloronly for Table 1')
            pred_ss, pred_sc, SSreport_color[rep,modelNumber - 1], SCreport_color[
            rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_coloronly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_color[rep,modelNumber - 1], CSreport_color[
            rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_coloronly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_colorC, clf_colorS)


            BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                                BP_layer2_out, 1,
                                                                                                'noskip')  # bp retrievals from layer 1
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            # Table 1 (memory retrievals from L1)
            print('classifiers accuracy for L1 ')
            pred_ss, pred_sc, SSreport_l1[rep,modelNumber - 1], SCreport_l1[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_l1[rep,modelNumber - 1], CSreport_l1[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                             bs_testing, clf_colorC, clf_colorS)


            BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                                BP_layer2_out, 2,                                                                                            'noskip')  # bp retrievals from layer 2
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()

            # Table 1 (memory retrievals from L2)
            print('classifiers accuracy for L2 ')
            pred_ss, pred_sc, SSreport_l2[rep,modelNumber - 1], SCreport_l2[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_l2[rep,modelNumber - 1], CSreport_l2[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                             bs_testing, clf_colorC, clf_colorS)

    SSreport_both=SSreport_both.reshape(1,-1)
    SSreport_both=SSreport_both.reshape(1,-1)
    SCreport_both=SCreport_both.reshape(1,-1)
    CCreport_both=CCreport_both.reshape(1,-1)
    CSreport_both=CSreport_both.reshape(1,-1)
    SSreport_shape=SSreport_shape.reshape(1,-1)
    SCreport_shape=SCreport_shape.reshape(1,-1)
    CCreport_shape=CCreport_shape.reshape(1,-1)
    CSreport_shape=CSreport_shape.reshape(1,-1)
    CCreport_color=CCreport_color.reshape(1,-1)
    CSreport_color=CSreport_color.reshape(1,-1)
    SSreport_color=SSreport_color.reshape(1,-1)
    SCreport_color=SCreport_color.reshape(1,-1)
    SSreport_l1=SSreport_l1.reshape(1,-1)
    SCreport_l1=SCreport_l1.reshape(1,-1)
    CCreport_l1=CCreport_l1.reshape(1,-1)
    CSreport_l1=CSreport_l1.reshape(1,-1)

    SSreport_l2= SSreport_l2.reshape(1,-1)
    SCreport_l2=SCreport_l2.reshape(1,-1)
    CCreport_l2= CCreport_l2.reshape(1,-1)
    CSreport_l2=CSreport_l2.reshape(1,-1)



    outputFile.write(
        'Table 2 both shape and color, accuracy of SS {0:.4g} SE{1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_both.mean(),
            SSreport_both.std()/math.sqrt(numModels*perms),
            SCreport_both.mean(),
            SCreport_both.std()/math.sqrt(numModels*perms),
            CCreport_both.mean(),
            CCreport_both.std()/math.sqrt(numModels*perms),
            CSreport_both.mean(),
            CSreport_both.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 shape only, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_shape.mean(),
            SSreport_shape.std()/math.sqrt(numModels*perms),
            SCreport_shape.mean(),
            SCreport_shape.std()/math.sqrt(numModels*perms),
            CCreport_shape.mean(),
            CCreport_shape.std()/math.sqrt(numModels*perms),
            CSreport_shape.mean(),
            CSreport_shape.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 color only, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g},\n accuracy of SS {4:.4g} SE {5:.4g},accuracy of SC {6:.4g} SE {7:.4g}\n'.format(
            CCreport_color.mean(),
            CCreport_color.std()/math.sqrt(numModels*perms),
            CSreport_color.mean(),
            CSreport_color.std()/math.sqrt(numModels*perms),
            SSreport_color.mean(),
            SSreport_color.std()/math.sqrt(numModels*perms),
            SCreport_color.mean(),
            SCreport_color.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 l1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l1.mean(),
            SSreport_l1.std()/math.sqrt(numModels*perms),
            SCreport_l1.mean(),
            SCreport_l1.std()/math.sqrt(numModels*perms),
            CCreport_l1.mean(),
            CCreport_l1.std()/math.sqrt(numModels*perms),
            CSreport_l1.mean(),
            CSreport_l1.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 l2, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l2.mean(),
            SSreport_l2.std()/math.sqrt(numModels*perms),
            SCreport_l2.mean(),
            SCreport_l2.std()/math.sqrt(numModels*perms),
            CCreport_l2.mean(),
            CCreport_l2.std()/math.sqrt(numModels*perms),
            CSreport_l2.mean(),
            CSreport_l2.std()/math.sqrt(numModels*perms)))



################################################ storing visual information (shape and color) along with the categorical label##########

if Tab1SuppFlag ==1:

    print('Table 1S computing the accuracy of storing labels along with shape and color information')

    ftest_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=False,transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]),download=False)
    ftest_dataset.targets= ftest_dataset.targets+10
    test_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=False,transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]),download=False)

    #build a combined dataset out of MNIST and Fasion MNIST
    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))


    perms = hugepermnum

    setSizes = [5,6,7,8]  # number of tokens

    numModels=10


    totalAccuracyShape = list()
    totalSEshape=list()
    totalAccuracyColor = list()
    totalSEcolor=list()
    totalAccuracyShape_visual=list()
    totalSEshapeVisual=list()
    totalAccuracyColor_visual=list()
    totalSEcolorVisual=list()
    totalAccuracyShapeWlabels = list()
    totalSEshapeWlabels=list()
    totalAccuracyColorWlabels = list()
    totalSEcolorWlabels=list()
    totalAccuracyShape_half=list()
    totalSEshape_half=list()
    totalAccuracyColor_half=list()
    totalSEcolor_half=list()
    totalAccuracyShape_cat=list()
    totalSEshape_cat=list()
    totalAccuracyColor_cat=list()
    totalSEcolor_cat=list()

    shape_dotplots_models=list() #data for the dot plots
    color_dotplots_models=list()
    shapeVisual_dotplots_models=list()
    colorVisual_dotplots_models=list()
    shapeWlabels_dotplots_models=list()
    colorWlabels_dotplots_models=list()
    shape_half_dotplots_models=list()
    color_half_dotplots_models=list()
    shape_cat_dotplots_models=list()
    color_cat_dotplots_models=list()


    for numItems in setSizes:
            print('Doing label/shape storage:  Setsize {num}'.format(num=numItems))
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=0)


            accuracyShapeModels=list()
            accuracyColorModels=list()
            accuracyShapeWlabelsModels=list()
            accuracyColorWlabelsModels=list()
            accuracyShapeVisualModels=list()
            accuracyColorVisualModels=list()
            accuracyShapeModels_half = list()
            accuracyColorModels_half = list()
            accuracyShapeModels_cat = list()
            accuracyColorModels_cat = list()



            for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10

                accuracyShape = list()
                accuracyColor = list()
                accuracyShape_visual=list()
                accuracyColor_visual=list()
                accuracyShape_wlabels = list()
                accuracyColor_wlabels = list()

                accuracyShape_half = list()
                accuracyColor_half = list()
                accuracyShape_cat = list()
                accuracyColor_cat = list()


                load_checkpoint(
                    'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

                print('doing model {0} for Table 1S'.format(modelNumber))
                clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
                clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
                clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
                clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


                #the ratio of visual information encoded into memory
                shape_coeff = 1
                color_coeff = 1
                shape_coeff_half=.5
                color_coeff_half=.5
                shape_coeff_cat = 0
                color_coeff_cat = 0

                for i in range(perms):
                    # print('iterstart')
                    images, shapelabels = next(iter(test_loader_smaller))  # load up a set of digits

                    imgs = images.view(-1, 3 * 28 * 28).cuda()
                    colorlabels = torch.round(
                        imgs[:, 0] * 255)

                    l1_act, l2_act, shape_act, color_act = activations(imgs)

                    shapepred, x, y, z = classifier_shapemap_test_imgs(shape_act, shapelabels, colorlabels, numItems,
                                                                       clf_shapeS, clf_shapeC)
                    colorpred, x, y, z = classifier_colormap_test_imgs(color_act, shapelabels, colorlabels, numItems,
                                                                       clf_colorC, clf_colorS)
                    # one hot coding of labels before storing into the BP
                    shape_onehot = F.one_hot(shapepred, num_classes=20)
                    shape_onehot = shape_onehot.float().cuda()
                    color_onehot = F.one_hot(colorpred, num_classes=10)
                    color_onehot = color_onehot.float().cuda()
                    #binding output when only maps are stored;  storeLabels=0
                    shape_out, color_out, L2_out, L1_out, shapelabel_junk, colorlabel_junk=BPTokens_with_labels(
                        bpsize, bpPortion, 0,shape_coeff, color_coeff, shape_act, color_act, l1_act, l2_act, shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)


                    shapepredVisual, x, ssreportVisual, z = classifier_shapemap_test_imgs(shape_out, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpredVisual, x, ccreportVisual, z = classifier_colormap_test_imgs(color_out, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)



                    #binding output that stores map activations + labels
                    shape_out_all, color_out_all, l2_out_all, l1_out_all, shape_label_out, color_label_out = BPTokens_with_labels(
                        bpsize, bpPortion, 1,shape_coeff, color_coeff, shape_act, color_act, l1_act, l2_act, shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred, x, ssreport, z = classifier_shapemap_test_imgs(shape_out_all, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred, x, ccreport, z = classifier_colormap_test_imgs(color_out_all, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel = shape_label_out.argmax(1)
                    retrievedcolorlabel = color_label_out.argmax(1)


                    # Compare accuracy against the original labels
                    accuracy_shape = torch.eq(shapelabels.cpu(), retrievedshapelabel.cpu()).sum().float() / numItems
                    accuracy_color = torch.eq(colorlabels.cpu(), retrievedcolorlabel.cpu()).sum().float() / numItems
                    accuracyShape.append(accuracy_shape)  # appends the perms
                    accuracyColor.append(accuracy_color)
                    accuracyShape_visual.append(ssreportVisual)
                    accuracyColor_visual.append(ccreportVisual)
                    accuracyShape_wlabels.append(ssreport)
                    accuracyColor_wlabels.append(ccreport)

                    # binding output that stores 50% of map activations + labels
                    shape_out_all_half, color_out_all_half, l2_out_all_half, l1_out_all_half, shape_label_out_half, color_label_out_half = BPTokens_with_labels(
                        bpsize, bpPortion, 1, shape_coeff_half, color_coeff_half, shape_act, color_act, l1_act, l2_act,
                        shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred_half, x, ssreport_half, z = classifier_shapemap_test_imgs(shape_out_all_half, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred_half, x, ccreport_half, z = classifier_colormap_test_imgs(color_out_all_half, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel_half = shape_label_out_half.argmax(1)
                    retrievedcolorlabel_half = color_label_out_half.argmax(1)

                    accuracy_shape_half = torch.eq(shapelabels.cpu(), retrievedshapelabel_half.cpu()).sum().float() / numItems
                    accuracy_color_half = torch.eq(colorlabels.cpu(), retrievedcolorlabel_half.cpu()).sum().float() / numItems
                    accuracyShape_half.append(accuracy_shape_half)  # appends the perms
                    accuracyColor_half.append(accuracy_color_half)


                      # binding output that stores only labels with 0% visual information
                    shape_out_all_cat, color_out_all_cat, l2_out_all_cat, l1_out_all_cat, shape_label_out_cat, color_label_out_cat = BPTokens_with_labels(
                        bpsize, bpPortion, 1, shape_coeff_cat, color_coeff_cat, shape_act, color_act, l1_act, l2_act,
                        shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred_cat, x, ssreport_cat, z = classifier_shapemap_test_imgs(shape_out_all_cat, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred_cat, x, ccreport_cat, z = classifier_colormap_test_imgs(color_out_all_cat, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel_cat = shape_label_out_cat.argmax(1)
                    retrievedcolorlabel_cat = color_label_out_cat.argmax(1)

                    accuracy_shape_cat = torch.eq(shapelabels.cpu(), retrievedshapelabel_cat.cpu()).sum().float() / numItems
                    accuracy_color_cat = torch.eq(colorlabels.cpu(), retrievedcolorlabel_cat.cpu()).sum().float() / numItems
                    accuracyShape_cat.append(accuracy_shape_cat)  # appends the perms
                    accuracyColor_cat.append(accuracy_color_cat)





                #append the accuracy for all models
                accuracyShapeModels.append(sum(accuracyShape) / perms)
                accuracyColorModels.append(sum(accuracyColor) / perms)
                accuracyShapeVisualModels.append(sum(accuracyShape_visual)/perms)
                accuracyColorVisualModels.append(sum(accuracyColor_visual)/perms)
                accuracyShapeWlabelsModels.append(sum(accuracyShape_wlabels) / perms)
                accuracyColorWlabelsModels.append(sum(accuracyColor_wlabels) / perms)
                accuracyShapeModels_half.append(sum(accuracyShape_half) / perms)
                accuracyColorModels_half.append(sum(accuracyColor_half) / perms)
                accuracyShapeModels_cat.append(sum(accuracyShape_cat) / perms)
                accuracyColorModels_cat.append(sum(accuracyColor_cat) / perms)


            shape_dotplots_models.append(torch.stack(accuracyShapeModels).view(1,-1))
            totalAccuracyShape.append(torch.stack(accuracyShapeModels).mean())
            totalSEshape.append(torch.stack(accuracyShapeModels).std()/math.sqrt(numModels))

            color_dotplots_models.append(torch.stack(accuracyColorModels).view(1,-1))
            totalAccuracyColor.append(torch.stack(accuracyColorModels).mean())
            totalSEcolor.append(torch.stack(accuracyColorModels).std() / math.sqrt(numModels))

            shapeVisual_dotplots_models.append(torch.stack(accuracyShapeVisualModels).view(1,-1))
            totalAccuracyShape_visual.append(torch.stack(accuracyShapeVisualModels).mean())
            totalSEshapeVisual.append(torch.stack(accuracyShapeVisualModels).std()/math.sqrt(numModels))

            colorVisual_dotplots_models.append(torch.stack(accuracyColorVisualModels).view(1,-1))
            totalAccuracyColor_visual.append(torch.stack(accuracyColorVisualModels).mean())
            totalSEcolorVisual.append(torch.stack(accuracyColorVisualModels).std() / math.sqrt(numModels))

            shapeWlabels_dotplots_models.append(torch.stack(accuracyShapeWlabelsModels).view(1,-1))
            totalAccuracyShapeWlabels .append(torch.stack(accuracyShapeWlabelsModels).mean())
            totalSEshapeWlabels.append(torch.stack(accuracyShapeWlabelsModels).std() / math.sqrt(numModels))

            colorWlabels_dotplots_models.append(torch.stack(accuracyColorWlabelsModels).view(1,-1))
            totalAccuracyColorWlabels.append(torch.stack(accuracyColorWlabelsModels).mean())
            totalSEcolorWlabels.append(torch.stack(accuracyColorWlabelsModels).std() / math.sqrt(numModels))

            shape_half_dotplots_models.append(torch.stack(accuracyShapeModels_half).view(1,-1))
            totalAccuracyShape_half.append(torch.stack(accuracyShapeModels_half).mean())
            totalSEshape_half.append(torch.stack(accuracyShapeModels_half).std() / math.sqrt(numModels))

            color_half_dotplots_models.append(torch.stack(accuracyColorModels_half).view(1,-1))
            totalAccuracyColor_half.append(torch.stack(accuracyColorModels_half).mean())
            totalSEcolor_half.append(torch.stack(accuracyColorModels_half).std() / math.sqrt(numModels))

            shape_cat_dotplots_models.append(torch.stack(accuracyShapeModels_cat).view(1,-1))
            totalAccuracyShape_cat.append(torch.stack(accuracyShapeModels_cat).mean())
            totalSEshape_cat.append(torch.stack(accuracyShapeModels_cat).std() / math.sqrt(numModels))

            color_cat_dotplots_models.append(torch.stack(accuracyColorModels_cat).view(1,-1))
            totalAccuracyColor_cat.append(torch.stack(accuracyColorModels_cat).mean())
            totalSEcolor_cat.append(torch.stack(accuracyColorModels_cat).std() / math.sqrt(numModels))







    print(shape_dotplots_models)
    print(color_dotplots_models)
    print(shapeVisual_dotplots_models)
    print(colorVisual_dotplots_models)
    print(shapeWlabels_dotplots_models)
    print(colorWlabels_dotplots_models)
    print(shape_half_dotplots_models)
    print(color_half_dotplots_models)
    print(shape_cat_dotplots_models)
    print(color_cat_dotplots_models)

    outputFile.write('Table 1S, accuracy of ShapeLabel')


    for i in range(len(setSizes)):
        outputFile.write('\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape[i],totalSEshape[i] ))

    outputFile.write('\n\nTable 3, accuracy of ColorLabel')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor[i], totalSEcolor[i]))

    outputFile.write('\n\nTable 3, accuracy of shape map with no labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_visual[i], totalSEshapeVisual[i]))

    outputFile.write('\n\nTable 3, accuracy of color map with no labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_visual[i],
                                                                 totalSEcolorVisual[i]))

    outputFile.write('\n\nTable 3, accuracy of shape map with labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShapeWlabels[i],
                                                                totalSEshapeWlabels[i]))

    outputFile.write('\n\nTable 3, accuracy of color map with labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColorWlabels[i],
                                                                 totalSEcolorWlabels[i]))

    outputFile.write('\n\nTable 3, accuracy of ShapeLabel for 50% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_half[i], totalSEshape_half[i]))

    outputFile.write('\n\nTable 3, accuracy of ColorLabel for 50% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_half[i], totalSEcolor_half[i]))

    outputFile.write('\n\nTable 3, accuracy of ShapeLabel for 0% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_cat[i], totalSEshape_cat[i]))

    outputFile.write('\n\nTable 3, accuracy of ColorLabel for 0% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_cat[i], totalSEcolor_cat[i]))

if TabTwoColorFlag1 == 1:
    modelNumber=1

    print('doing model {0} for Table 1'.format(modelNumber))
    clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
    clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
    clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
    clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


    print('Doing Table 1')
    numimg = 10
    trans2 = transforms.ToTensor()
    imgs_all = []
    convert_tensor = transforms.ToTensor()
    convert_image = transforms.ToPILImage()
    #for i in range (1,numimg+1):
    img = Image.open('{each}_thick.png'.format(each=9))
    img = convert_tensor(img)
    img_new = img[0:3,:,:]*1  #Ian

    imgs_all.append(img_new)
    imgs_all = torch.stack(imgs_all)
    imgs = imgs_all.view(-1, 3 * 28 * 28).cuda()
    img_new = convert_image(img_new)
    save_image(
            torch.cat([trans2(img_new).view(1, 3, 28, 28)], 0),
            'output{num}/figure10test.png'.format(num=modelNumber),
            nrow=numimg,
            normalize=False,
            range=(-1, 1),
        )

            # run them all through the encoder
    l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

    # now store and retrieve them from the BP
    BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                 shape_act, color_act,
                                                                                 shape_coeff, color_coeff, 1, 1,
                                                                                 normalize_fact_familiar)
    BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act,
                                                                                           shape_coeff, 0, 0, 0,
                                                                                           normalize_fact_familiar)
    BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act, 0,
                                                                                           color_coeff, 0, 0,
                                                                                           normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0,
                                                                                l1_coeff, 0,
                                                                                normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0, 0,
                                                                                l2_coeff,
                                                                                normalize_fact_familiar)
    #print(clf_colorC, clf_colorS)
    pred_cc,pred_cs,pred_cb = classifier_color_prediction(clf_colorC,color_out_BP_coloronly, color_out_BP_shapeonly,color_out_BP_both)
    #outputFile.write(f'Color from color:{pred_cc}, Color from shape:{pred_cs}')# Color from both:{pred_cb}')
    print(pred_cc)
if TabTwoColorFlag == 1:
    numModels=1
    perms=1
    SSreport_both = np.tile(0.0, [perms,numModels])
    SCreport_both = np.tile(0.0, [perms,numModels])
    CCreport_both = np.tile(0.0, [perms,numModels])
    CSreport_both = np.tile(0.0, [perms,numModels])
    SSreport_shape = np.tile(0.0, [perms,numModels])
    SCreport_shape = np.tile(0.0, [perms,numModels])
    CCreport_shape = np.tile(0.0, [perms,numModels])
    CSreport_shape = np.tile(0.0, [perms,numModels])
    SSreport_color = np.tile(0.0, [perms,numModels])
    SCreport_color = np.tile(0.0, [perms,numModels])
    CCreport_color = np.tile(0.0, [perms,numModels])
    CSreport_color = np.tile(0.0, [perms,numModels])
    SSreport_l1 = np.tile(0.0, [perms,numModels])
    SCreport_l1= np.tile(0.0, [perms,numModels])
    CCreport_l1 = np.tile(0.0, [perms,numModels])
    CSreport_l1 = np.tile(0.0, [perms,numModels])
    SSreport_l2 = np.tile(0.0, [perms,numModels])
    SCreport_l2 = np.tile(0.0, [perms,numModels])
    CCreport_l2 = np.tile(0.0, [perms,numModels])
    CSreport_l2 = np.tile(0.0, [perms,numModels])


    #for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
    #load_checkpoint(
            #'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))
    modelNumber=1

    print('doing model {0} for Table 1'.format(modelNumber))
    clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
    clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
    clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
    clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))


    print('Doing Table 1')
    numimg = 10
    imgs_all = []
    trans2 = transforms.ToTensor()

    convert_tensor = transforms.ToTensor()
    convert_image = transforms.ToPILImage()
    for i in range (1,numimg+1):
        img = Image.open('{each}_thick.png'.format(each=i))
        img = convert_tensor(img)
        img_new = img[0:3,:,:]*1   #Ian

        imgs_all.append(img_new)
    imgs_all = torch.stack(imgs_all)
    imgs = imgs_all.view(-1, 3 * 28 * 28).cuda()
    img_new = convert_image(img_new)
    save_image(
            torch.cat([trans2(img_new).view(1, 3, 28, 28)], 0),
            'output{num}/figure10test.png'.format(num=modelNumber),
            nrow=numimg,
            normalize=False,
            range=(-1, 1),
        )
    #slice labels to prevent an error
    colorlabels = colorlabels[0:numimg]
    shapelabels = shapelabels[0:numimg]
    print(colorlabels)
    print(shapelabels)
            # run them all through the encoder
    l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

    # now store and retrieve them from the BP
    BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                 shape_act, color_act,
                                                                                 shape_coeff, color_coeff, 1, 1,
                                                                                 normalize_fact_familiar)
    BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act,
                                                                                           shape_coeff, 0, 0, 0,
                                                                                           normalize_fact_familiar)
    BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                           l2_act, shape_act,
                                                                                           color_act, 0,
                                                                                           color_coeff, 0, 0,
                                                                                           normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0,
                                                                                l1_coeff, 0,
                                                                                normalize_fact_familiar)

    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                shape_act, color_act, 0, 0, 0,
                                                                                l2_coeff,
                                                                                normalize_fact_familiar)

    # Table 1: classifier accuracy for shape and color for memory retrievals
    print('classifiers accuracy for memory retrievals of BN_both for Table 1')
    pred_ss, pred_sc, SSreport_both[rep,modelNumber-1], SCreport_both[rep,modelNumber-1] = classifier_shapemap_test_imgs(shape_out_BP_both, shapelabels,
                                                                     colorlabels, bs_testing, clf_shapeS,
                                                                     clf_shapeC)
    pred_cc, pred_cs, CCreport_both[rep,modelNumber-1], CSreport_both[rep,modelNumber-1]= classifier_colormap_test_imgs(color_out_BP_both, shapelabels,
                                                                     colorlabels, bs_testing, clf_colorC,clf_colorS)



    # Table 1: classifier accuracy for shape and color for memory retrievals
    print('classifiers accuracy for memory retrievals of BN_shapeonly for Table 1')
    pred_ss, pred_sc, SSreport_shape[rep,modelNumber - 1], SCreport_shape[
    rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_shapeonly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_shape[rep,modelNumber - 1], CSreport_shape[
    rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_shapeonly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_colorC, clf_colorS)

    print('classifiers accuracy for memory retrievals of BN_coloronly for Table 1')
    pred_ss, pred_sc, SSreport_color[rep,modelNumber - 1], SCreport_color[
    rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_coloronly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_color[rep,modelNumber - 1], CSreport_color[
    rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_coloronly, shapelabels,
                                                     colorlabels,
                                                     bs_testing, clf_colorC, clf_colorS)


    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                        BP_layer2_out, 1,
                                                                                        'noskip')  # bp retrievals from layer 1
    z_color = vae.sampling(mu_color, log_var_color).cuda()
    z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
    # Table 1 (memory retrievals from L1)
    print('classifiers accuracy for L1 ')
    
    pred_ss, pred_sc, SSreport_l1[rep,modelNumber - 1], SCreport_l1[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_l1[rep,modelNumber - 1], CSreport_l1[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                     bs_testing, clf_colorC, clf_colorS)


    BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                        BP_layer2_out, 2,                                                                                            'noskip')  # bp retrievals from layer 2
    z_color = vae.sampling(mu_color, log_var_color).cuda()
    z_shape = vae.sampling(mu_shape, log_var_shape).cuda()

    # Table 1 (memory retrievals from L2)
    print('classifiers accuracy for L2 ')
    pred_ss, pred_sc, SSreport_l2[rep,modelNumber - 1], SCreport_l2[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                     bs_testing, clf_shapeS, clf_shapeC)
    pred_cc, pred_cs, CCreport_l2[rep,modelNumber - 1], CSreport_l2[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                     bs_testing, clf_colorC, clf_colorS)
                                                                     
    SSreport_both=SSreport_both.reshape(1,-1)
    SSreport_both=SSreport_both.reshape(1,-1)
    SCreport_both=SCreport_both.reshape(1,-1)
    CCreport_both=CCreport_both.reshape(1,-1)
    CSreport_both=CSreport_both.reshape(1,-1)
    SSreport_shape=SSreport_shape.reshape(1,-1)
    SCreport_shape=SCreport_shape.reshape(1,-1)
    CCreport_shape=CCreport_shape.reshape(1,-1)
    CSreport_shape=CSreport_shape.reshape(1,-1)
    CCreport_color=CCreport_color.reshape(1,-1)
    CSreport_color=CSreport_color.reshape(1,-1)
    SSreport_color=SSreport_color.reshape(1,-1)
    SCreport_color=SCreport_color.reshape(1,-1)
    SSreport_l1=SSreport_l1.reshape(1,-1)
    SCreport_l1=SCreport_l1.reshape(1,-1)
    CCreport_l1=CCreport_l1.reshape(1,-1)
    CSreport_l1=CSreport_l1.reshape(1,-1)

    SSreport_l2= SSreport_l2.reshape(1,-1)
    SCreport_l2=SCreport_l2.reshape(1,-1)
    CCreport_l2= CCreport_l2.reshape(1,-1)
    CSreport_l2=CSreport_l2.reshape(1,-1)



    outputFile.write(
        'TableTwoColor both shape and color, accuracy of SS {0:.4g} SE{1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_both.mean(),
            SSreport_both.std()/math.sqrt(numModels*perms),
            SCreport_both.mean(),
            SCreport_both.std()/math.sqrt(numModels*perms),
            CCreport_both.mean(),
            CCreport_both.std()/math.sqrt(numModels*perms),
            CSreport_both.mean(),
            CSreport_both.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor shape only, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_shape.mean(),
            SSreport_shape.std()/math.sqrt(numModels*perms),
            SCreport_shape.mean(),
            SCreport_shape.std()/math.sqrt(numModels*perms),
            CCreport_shape.mean(),
            CCreport_shape.std()/math.sqrt(numModels*perms),
            CSreport_shape.mean(),
            CSreport_shape.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor color only, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g},\n accuracy of SS {4:.4g} SE {5:.4g},accuracy of SC {6:.4g} SE {7:.4g}\n'.format(
            CCreport_color.mean(),
            CCreport_color.std()/math.sqrt(numModels*perms),
            CSreport_color.mean(),
            CSreport_color.std()/math.sqrt(numModels*perms),
            SSreport_color.mean(),
            SSreport_color.std()/math.sqrt(numModels*perms),
            SCreport_color.mean(),
            SCreport_color.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor l1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l1.mean(),
            SSreport_l1.std()/math.sqrt(numModels*perms),
            SCreport_l1.mean(),
            SCreport_l1.std()/math.sqrt(numModels*perms),
            CCreport_l1.mean(),
            CCreport_l1.std()/math.sqrt(numModels*perms),
            CSreport_l1.mean(),
            CSreport_l1.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'TableTwoColor l2, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l2.mean(),
            SSreport_l2.std()/math.sqrt(numModels*perms),
            SCreport_l2.mean(),
            SCreport_l2.std()/math.sqrt(numModels*perms),
            CCreport_l2.mean(),
            CCreport_l2.std()/math.sqrt(numModels*perms),
            CSreport_l2.mean(),
            CSreport_l2.std()/math.sqrt(numModels*perms)))


######This part is to detect whether a stimulus is novel or familiar

if noveltyDetectionFlag==1:


    perms=smallpermnum
    numModels=10

    acc_fam=torch.zeros(numModels,perms)
    acc_nov=torch.zeros(numModels,perms)

    for modelNumber in range (1,numModels+1):

        acc_fam[modelNumber-1,:], acc_nov[modelNumber-1,:]= novelty_detect( perms, bpsize, bpPortion, shape_coeff, color_coeff, normalize_fact_familiar,
                  normalize_fact_novel, modelNumber, test_loader_smaller)
    mean_fam=acc_fam.view(1,-1).mean()
    fam_SE= acc_fam.view(1,-1).std()/(len(acc_fam.view(1,-1)))

    mean_nov=acc_nov.view(1,-1).mean()
    nov_SE= acc_nov.view(1,-1).std()/(len(acc_nov.view(1,-1)))





    outputFile.write(
            '\accuracy of detecting the familiar shapes : mean is {0:.4g} and SE is {1:.4g} '.format(mean_fam, fam_SE))

    outputFile.write(
            '\naccuracy of detecting the novel shapes : mean is {0:.4g} and SE is {1:.4g} '.format(mean_nov, nov_SE))

outputFile.close()

def plotbpvals(set1,set2,set3,set4,set5,label):

    #plot the values of the BP nodes..  this should be made into a function
    plt.figure()
    plt.subplot(151)
    plt.hist(set1, 100)
    plt.xlim(-10, 10)
    plt.subplot(152)
    plt.hist(set2, 100)
    plt.xlim(-10,10)
    plt.subplot(153)
    plt.hist(set3, 100)
    plt.subplot(154)
    plt.hist(set4, 100)
    plt.ylabel(label)
    plt.subplot(155)
    plt.hist(set5, 100)
    plt.show()     #visualize the values of the BP as a distribution
