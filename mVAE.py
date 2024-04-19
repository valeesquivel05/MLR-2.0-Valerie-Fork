#MLR 2.0

#The second installment of the MLR model line, written largely by Ian Deal and Brad Wyble
#This original version of this model is published in
#Hedayati, S., O’Donnell, R. E., & Wyble, B. (2022). A model of working memory for latent representations. Nature Human Behaviour, 6(5), 709-719.
#And the code in that work is a variant of
# MNIST VAE from http://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb
# Modified by Brad Wyble, Shekoo Hedayati

#In this version, the model adds to the original MLR model the following features:
#-a large Retina  (100 pixels wide)
#-Convolutional encoder and decoder
#-Location latent space  (in the horizontal diection)
#-improved loss functions for shape and color
#-White is now one of the 10 colors
#-Skip connection trained on bi-color stimuli
#-Label networks  akin to SVRHM paper:
#Hedayati, S., Beaty, R., & Wyble, B. (2021). Seeking the Building Blocks of Visual Imagery and Creativity in a Cognitively Inspired Neural Network. arXiv preprint arXiv:2112.06832.



# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os
from torch.utils.data import DataLoader, Subset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

# load a saved vae checkpoint
def load_checkpoint(filepath):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    return vae

global colorlabels
numcolors = 0

colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "white"]
colorlabels = np.random.randint(0, 10, 1000000)
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [1-colorrange,1-colorrange,1-colorrange]
]


#comment this
def Colorize_func(img):
    global numcolors,colorlabels

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')
    return img

#comment
def Colorize_func_specific(col,img):
    # col: an int index for which base color is being used
    rgb = colorvals[col]  # grab the rgb for this base color
    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')

    return img

# model training data set and dimensions
data_set_flag = 'padded_emnist' # mnist, cifar10, padded_mnist, padded_cifar10
imgsize = 28
retina_size = 100 # by default should be same size as image
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = retina_size * imgsize * 3
h_dim1 = 256
h_dim2 = 128
z_dim = 16
l_dim = 100

# must call the dataset_builder function from a seperate .py file
#to be moved
# padded cifar10, 2d retina, sight word latent,


#CNN VAE
#this model takes in a single cropped image and a location 1-hot vector  (to be replaced by an attentional filter that determines location from a retinal image)
#there are three latent spaces:location, shape and color and 6 loss functions
#loss functions are: shape, color, location, retinal, cropped (shape + color combined), skip

class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim):
        super(VAE_CNN, self).__init__()
        # encoder part
        self.l_dim = l_dim
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)  # Latent vectors mu and sigma

        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4) * 16, h_dim2)
        self.fc_bn2 = nn.BatchNorm1d(h_dim2) # remove
        # bottle neck part
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        self.fc35 = nn.Linear(l_dim, z_dim)  # location
        self.fc36 = nn.Linear(l_dim, z_dim)
        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc4l = nn.Linear(z_dim, l_dim)  # location
        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)
        self.fc8 = nn.Linear(h_dim2, h_dim2)

        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)
        # combine recon and location into retina

        self.fc6 = nn.Linear((imgsize+z_dim) * 3, (retina_size) * 6)
        self.fc65 = nn.Linear((retina_size) * 6, (retina_size) * 6) #recurrence layer
        self.fc7 = nn.Linear((retina_size) * 6, retina_size * 3)
        self.relu = nn.ReLU()
        self.skipconv = nn.Conv2d(16,16,kernel_size=1,stride=1,padding =0,bias=False)

        # map scalars
        self.shape_scale = 1.9
        self.color_scale = 1.8

    def encoder(self, x, l):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
        h = self.relu(self.fc_bn2(self.fc2(h)))
        hskip = self.fc8(h) # skip con fc2 to fc5

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), self.fc35(l), self.fc36(l), hskip # mu, log_var

    def location_encoder(self, l):
        return self.sampling_location(self.fc35(l), self.fc36(l))

    def sampling_location(self, mu, log_var):
        std = (0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_retinal(self, z_shape, z_color, z_location, hskip = None, whichdecode = None): #recurrent
        # digit recon
        if whichdecode == 'shape':
            h = (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        elif whichdecode == 'color':
            h = (F.relu(self.fc4c(z_color)) * self.color_scale)
        else:
            h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).detach().view(-1, 3, imgsize, imgsize) #detach conv
        h = torch.sigmoid(h)
        # location vector recon
        l = z_location.detach() #cont. repr of location
        l = l.view(-1,1,1,self.z_dim)
        l = torch.sigmoid(l)
        l = l.expand(-1, 3, imgsize, self.z_dim) # reshape to concat
        # combine into retina
        h = torch.cat([h,l], dim = 3)
        b_dim = h.size()[0]*h.size()[2]
        h = h.view(b_dim,-1)
        h = self.relu(self.fc6(h))
        h = self.relu(self.fc65(h)) #recurrence
        h = self.relu(self.fc65(h))
        h = self.fc7(h).view(-1,3,imgsize,retina_size)
        return torch.sigmoid(h)

    def decoder_color(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) * self.color_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_shape(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4s(z_shape)) * self.shape_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_location(self, z_shape, z_color, hskip, z_location):
        h = self.fc4l(z_location)
        return torch.sigmoid(h)

    def decoder_cropped(self, z_shape, z_color, z_location, hskip=0):
        h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_skip_cropped(self, z_shape, z_color, z_location, hskip):
        h = F.relu(hskip)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_skip_retinal(self, z_shape, z_color, z_location, hskip):
        # digit recon
        h = F.relu(hskip)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize).detach()
        h = torch.sigmoid(h)
        # location vector recon
        l = z_location.detach() #cont. repr of location
        l = l.view(-1,1,1,8)
        l = torch.sigmoid(l)
        l = l.expand(-1, 3, imgsize, 8) # reshape to concat
        # combine into retina
        h = torch.cat([h,l], dim = 3)
        b_dim = h.size()[0]*h.size()[2]
        h = h.view(b_dim,-1)
        h = self.relu(self.fc6(h))
        h = self.fc7(h).view(-1,3,imgsize,retina_size)
        return torch.sigmoid(h)

    def activations(self, z_shape, z_color, z_location):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape)) + F.relu(self.fc4l(z_location))
        fc4c = self.fc4c(z_color)
        fc4s = self.fc4s(z_shape)
        fc4l = self.fc4l(z_location)
        fc5 = self.fc5(h)
        return fc4c, fc4s, fc4l, fc5

    def forward_layers(self, l1, l2, layernum, whichdecode):
        if layernum == 1:
            h = F.relu(self.bn2(self.conv2(l1)))
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 2:
            h = self.relu(self.bn3(self.conv3(l2)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 3:
            hskip = self.fc8(l1)
            mu_shape = self.fc31(l1)
            log_var_shape = self.fc32(l1)
            mu_color = self.fc33(l1)
            log_var_color = self.fc34(l1)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape, z_color, 0, hskip)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(z_shape, z_color, 0, hskip)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, whichdecode='noskip', keepgrad=[]):
        if type(x) == list or type(x) == tuple:    #passing in a cropped+ location as input
            l = x[2].cuda()
            x = x[1].cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, hskip = self.encoder(x, l)
        else:  #passing in just cropped image
            x = x.cuda()
            l = torch.zeros(x.size()[0], self.l_dim).cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, hskip = self.encoder(x, l)

        #what maps are used in the training process.. the others are detached to zero out those gradients
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()

        if ('location' in keepgrad):
            z_location = self.sampling_location(mu_location, log_var_location)
        else:
            z_location = self.sampling_location(mu_location, log_var_location).detach()

        if ('skip' in keepgrad):
            hskip = hskip
        else:
            hskip = hskip.detach()

        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape,z_color, z_location, hskip)
        elif (whichdecode == 'retinal'):
            output = self.decoder_retinal(z_shape,z_color, z_location, 0)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(0, 0, 0, hskip)
        elif (whichdecode == 'skip_retinal'):
            output = self.decoder_skip_retinal(0, 0, z_location, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color , 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape,0, 0)
        elif (whichdecode == 'location'):
            output = self.decoder_location(0, 0, 0, z_location)

        return output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location

# function to build an  actual model instance
def vae_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1, h_dim2 = h_dim2, z_dim = z_dim, l_dim = l_dim):
    vae = VAE_CNN(x_dim, h_dim1, h_dim2, z_dim, l_dim)

    folder_path = f'sample_{vae_type}_{data_set_flag}'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return vae, z_dim

########Actually build it
vae, z_dim = vae_builder()

#######what optimier to use:
# learning rate = 0.0001
#optimizer = torch.optim.SGD(vae.parameters(), lr=0.0001, momentum = 0.9)
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

vae.cuda()

######the loss functions
#Pixelwise loss for the entire retina (dimensions are cropped image height x retina_size)
def loss_function(recon_x, x, mu, log_var, mu_c, log_var_c):
    x = x[0].clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, retina_size * imgsize * 3), reduction='sum')
    return BCE

#pixelwise loss for just the cropped image
def loss_function_crop(recon_x, x, mu, log_var, mu_c, log_var_c):
    x = x.clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, retina_size * imgsize * 3), reduction='sum')
    return BCE


# loss for shape in a cropped image
def loss_function_shape(recon_x, x, mu, log_var):
    x = x[1].clone().cuda()
    # make grayscale reconstruction
    gray_x = x.view(-1, 3, imgsize, imgsize).mean(1)
    gray_x = torch.stack([gray_x, gray_x, gray_x], dim=1)
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), gray_x.view(-1,imgsize * imgsize * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD

#loss for just color in a cropped image
def loss_function_color(recon_x, x, mu, log_var):
    x = x[1].clone().cuda()
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(-1, 3, imgsize * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, imgsize * imgsize * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just location
def loss_function_location(recon_x, x, mu, log_var):
    x = x[2].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# test recreate img with different features
def progress_out(data, epoch, count, skip = False, filename = None):
    sample_size = 25
    vae.eval()
    #make a filename if none is provided
    if filename == None:
        filename = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png'

    if skip:
        sample = data[:sample_size]
        with torch.no_grad():
            shape_color_dim = imgsize
            reconds, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'skip_cropped') #digit from skip
            recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
            reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
            recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape
            utils.save_image(
            torch.cat([sample.view(sample_size, 3, imgsize, shape_color_dim).cuda(), reconds.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recond.view(sample_size, 3, imgsize, shape_color_dim).cuda(),
                    reconc.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recons.view(sample_size, 3, imgsize, shape_color_dim).cuda()], 0),
            filename,
            nrow=sample_size, normalize=False, range=(-1, 1),)

    else:
        sample_data = data[0]
        sample_data[0] = sample_data[0][:sample_size]
        sample_data[1] = sample_data[1][:sample_size]
        sample_data[2] = sample_data[2][:sample_size]
        sample = sample_data
        with torch.no_grad():
            reconl, mu_color, log_var_color, mu_shape, log_var_shape,mu_location, log_var_location = vae(sample, 'location') #location
            reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'retinal') #retina
            recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
            reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
            recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape

        empty_retina = torch.zeros((sample_size, 3, imgsize, retina_size))

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
            n_reconc[i][0, :, 0:imgsize] = reconc[i][0]
            n_reconc[i][1, :, 0:imgsize] = reconc[i][1]
            n_reconc[i][2, :, 0:imgsize] = reconc[i][2]

        n_recons = empty_retina.clone()
        for i in range(len(recons)):
            n_recons[i][0, :, 0:imgsize] = recons[i][0]
            n_recons[i][1, :, 0:imgsize] = recons[i][1]
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
            torch.cat([sample.view(sample_size, 3, imgsize, shape_color_dim), reconb.view(sample_size, 3, imgsize, shape_color_dim), n_recond.view(sample_size, 3, imgsize, shape_color_dim), n_reconl.view(sample_size, 3, imgsize, shape_color_dim),
                    n_reconc.view(sample_size, 3, imgsize, shape_color_dim), n_recons.view(sample_size, 3, imgsize, shape_color_dim)], 0),
            filename,
            nrow=sample_size, normalize=False, range=(-1, 1),)

def test_loss(test_data, whichdecode = []):
    loss_dict = {}

    for decoder in whichdecode:
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(test_data, decoder)
        
        if decoder == 'retinal':
            loss = loss_function(recon_batch, test_data, mu_shape, log_var_shape, mu_color, log_var_color)
        
        elif decoder == 'cropped':
            loss = loss_function_crop(recon_batch, test_data[1], mu_shape, log_var_shape, mu_color, log_var_color)
        
        loss_dict[decoder] = loss.item()

    return loss_dict

def train(epoch, train_loader_noSkip, emnist_skip, fmnist_skip, test_loader, return_loss = False):
    vae.train()
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) # the latent space is trained on EMNIST, MNIST, and f-MNIST
    m = 6 # number of seperate training decoders used
    dataiter_emnist_skip= iter(emnist_skip) # The skip connection is trained on pairs from EMNIST, MNIST, and f-MNIST composed on top of each other
    dataiter_fmnist_skip= iter(fmnist_skip)
    test_iter = iter(test_loader)
    count = 0
    max_iter = 800
    loader=tqdm(train_loader_noSkip, total = max_iter)

    retinal_loss_train, cropped_loss_train = 0, 0 # loss metrics returned to training.py
    
    for i,j in enumerate(loader):
        data_noSkip = dataiter_noSkip.next()
        if count % 9 in [0,1,2,4,5]:
            data_skip = dataiter_emnist_skip.next()
        else:
            data_skip = dataiter_fmnist_skip.next()
        
        data = data_noSkip[0]

        count += 1
        optimizer.zero_grad()

        if epoch > 45: # increase the number of times retinal/location is trained
           m = 8
        
        if count% m == 0:
            whichdecode_use = 'shape'
            keepgrad = ['shape']

        elif count% m == 1:
            whichdecode_use = 'color'
            keepgrad = ['color']

        elif count% m in [2,7]:
            whichdecode_use = 'location'
            keepgrad = ['location']

        elif count% m in [3,6]:
            whichdecode_use = 'retinal'
            keepgrad = []

        elif count% m == 4:
            data = data[1] + data_skip[0]
            whichdecode_use = 'skip_cropped'
            keepgrad = ['skip']

        else:
            whichdecode_use = 'cropped'
            keepgrad = ['shape', 'color']

        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(data, whichdecode_use, keepgrad)
            
        if whichdecode_use == 'shape':  # shape
            loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)
            loss.backward()

        elif whichdecode_use == 'color': # color
            loss = loss_function_color(recon_batch, data, mu_color, log_var_color)
            loss.backward()

        elif whichdecode_use == 'location': # location
            loss = loss_function_location(recon_batch, data, mu_location, log_var_location)
            loss.backward()

        elif whichdecode_use == 'retinal': # retinal
            loss = loss_function(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
            loss.backward()
            retinal_loss_train = loss.item()

        elif whichdecode_use == 'cropped': # cropped
            loss = loss_function_crop(recon_batch, data[1], mu_shape, log_var_shape, mu_color, log_var_color)
            loss.backward()
            cropped_loss_train = loss.item()

        elif whichdecode_use == 'skip_cropped': # skip training
            loss = loss_function_crop(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
            loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description(
            (
                f'epoch: {epoch}; mse: {loss.item():.5f};'
            )
        )
        if count % 400 == 0:
            progress_out(data_noSkip, epoch, count)
        elif count % 500 == 0:
            data = data_noSkip[0][1] + data_skip[0]
            progress_out(data, epoch, count, skip= True)
        
        if i == max_iter +1:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))
    
    if return_loss is True:
        # get test losses for cropped and retinal
        test_data = test_iter.next()
        test_data = test_data[0]

        test_loss_dict = test_loss(test_data, ['retinal', 'cropped'])
    
        return [retinal_loss_train, test_loss_dict['retinal'], cropped_loss_train, test_loss_dict['cropped']]

#compute avg loss of retinal recon w/ skip, w/o skip, increase fc?
def test(whichdecode, test_loader_noSkip, test_loader_skip, bs):
    vae.eval()
    global numcolors
    test_loss = 0
    testiter_noSkip = iter(test_loader_noSkip)  # the latent space is trained on MNIST and f-MNIST
    testiter_skip = iter(test_loader_skip)  # The skip connection is trained on notMNIST
    with torch.no_grad():
        for i in range(1, len(test_loader_noSkip)): # get the next batch


            data = testiter_noSkip.next()
            data = data[0]
            recon, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(data, 'retinal')

            # sum up batch loss
            #test_loss += loss_function_shape(recon, data, mu_shape, log_var_shape).item()
            #test_loss += loss_function_color(recon, data, mu_color, log_var_color).item()
            test_loss += loss_function(recon, data, mu_shape, log_var_shape, mu_color, log_var_color).item()

    print('Example reconstruction')
    datac = data[0].cuda()
    datac=datac.view(bs, 3, imgsize, retina_size)
    save_image(datac[0:8], f'{args.dir}/orig.png')
    pos = torch.zeros((64,100)).cuda()
    for i in range(len(pos)):
        pos[i][random.randint(0,99)] = 1
    pos_mu = vae.fc35(pos)
    pos_logvar = vae.fc36(pos)

    # current imagining of shape and color results in random noise
    # generate a
    print('Imagining a shape')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 0
        zs = torch.randn(64, z_dim).cuda() * 1
        zl = vae.sampling(pos_mu, pos_logvar)
        sample = vae.decoder_retinal(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/sampleshape.png')


    print('Imagining a color')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 1
        zs = torch.randn(64, z_dim).cuda() * 0
        zl = vae.sampling(pos_mu, pos_logvar)
        sample = vae.decoder_retinal(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/samplecolor.png')

    test_loss /= len(test_loader_noSkip.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def activations(image, l= None):
    if l is None:
        l = torch.zeros(image.size()[0], vae.l_dim).cuda()

    h = vae.relu(vae.bn1(vae.conv1(image)))
    h = vae.relu(vae.bn2(vae.conv2(h)))
    h = vae.relu(vae.bn3(vae.conv3(h)))
    h = vae.relu(vae.bn4(vae.conv4(h)))
    h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
    l1_act = vae.relu(vae.fc_bn2(vae.fc2(h))) # fc after conv, drives skip connection
    l1_actf = F.relu(vae.bn1(vae.conv1(image))) # input layer
    l2_act = F.relu(vae.bn2(vae.conv2(l1_actf)))

    mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, hskip = vae.encoder(image, l)
    shape_act = vae.sampling(mu_shape, log_var_shape)
    color_act = vae.sampling(mu_color, log_var_color)
    location_act = vae.sampling_location(mu_location, log_var_location)
    return l1_act, l2_act, shape_act, color_act, location_act

def image_activations(image, l = None):
    if l is None:
        l = torch.zeros(image.size()[0], vae.l_dim).cuda()
    mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, hskip = vae.encoder(image, l)
    shape_act = vae.sampling(mu_shape, log_var_shape)
    color_act = vae.sampling(mu_color, log_var_color)
    location_act = vae.sampling_location(mu_location, log_var_location)
    return shape_act, color_act, location_act

def activation_fromBP(L1_activationBP, L2_activationBP, layernum):
    if layernum == 1:
        l2_act_bp = F.relu(vae.fc2(L1_activationBP))
        mu_shape = (vae.fc31(l2_act_bp))
        log_var_shape = (vae.fc32(l2_act_bp))
        mu_color = (vae.fc33(l2_act_bp))
        log_var_color = (vae.fc34(l2_act_bp))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    elif layernum == 2:
        mu_shape = (vae.fc31(L2_activationBP))
        log_var_shape = (vae.fc32(L2_activationBP))
        mu_color = (vae.fc33(L2_activationBP))
        log_var_color = (vae.fc34(L2_activationBP))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    return shape_act_bp, color_act_bp

def BPTokens_storage(bpsize, bpPortion,l1_act, l2_act, shape_act, color_act, location_act, shape_coeff, color_coeff, location_coeff, l1_coeff,l2_coeff, bs_testing, normalize_fact):
    notLink_all = list()  # will be used to accumulate the specific token linkages
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    tokenBindings = list()
    bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
    bp_in_color_dim = color_act.shape[1]
    bp_in_location_dim = location_act.shape[1]
    bp_in_L1_dim = l1_act.shape[1]
    bp_in_L2_dim = l2_act.shape[1]
    shape_fw = torch.randn(bp_in_shape_dim,
                            bpsize).cuda()  # make the randomized fixed weights to the binding pool
    color_fw = torch.randn(bp_in_color_dim, bpsize).cuda()
    location_fw = torch.randn(bp_in_color_dim, bpsize).cuda()
    L1_fw = torch.randn(bp_in_L1_dim, bpsize).cuda()
    L2_fw = torch.randn(bp_in_L2_dim, bpsize).cuda()

    # ENCODING!  Store each item in the binding pool
    for items in range(bs_testing):  # the number of images
        tkLink_tot = torch.randperm(bpsize)  # for each token figure out which connections will be set to 0
        notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

        BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coeff + torch.mm(
        color_act[items, :].view(1, -1), color_fw) * color_coeff + torch.mm(
        location_act[items, :].view(1, -1), location_fw) * location_coeff + torch.mm(
        l1_act[items, :].view(1, -1), L1_fw) * l1_coeff + torch.mm(l2_act[items, :].view(1, -1), L2_fw) * l2_coeff

        BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
        BP_in_all.append(BP_in_eachimg)  # appending and stacking images
        notLink_all.append(notLink)
    # now sum all of the BPs together to form one consolidated BP activation set.
    BP_in_items = torch.stack(BP_in_all)
    BP_in_items = torch.squeeze(BP_in_items, 1)
    BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # Add them up
    tokenBindings.append(torch.stack(notLink_all))  # this is the set of 0'd connections for each of the tokens
    tokenBindings.append(shape_fw)
    tokenBindings.append(color_fw)
    tokenBindings.append(location_fw)
    tokenBindings.append(L1_fw)
    tokenBindings.append(L2_fw)

    return BP_in_items, tokenBindings



def BPTokens_retrieveByToken( bpsize, bpPortion, BP_in_items,tokenBindings, l1_act, l2_act, shape_act, color_act, location_act,bs_testing,normalize_fact):
# NOW REMEMBER THE STORED ITEMS
    #notLink_all = list()  # will be used to accumulate the specific token linkages
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    notLink_all = tokenBindings[0]
    shape_fw = tokenBindings[1]
    color_fw = tokenBindings[2]
    location_fw = tokenBindings[3]
    L1_fw = tokenBindings[4]
    L2_fw = tokenBindings[5]
   
    tokenBindings.append(shape_fw)
    tokenBindings.append(color_fw)
    tokenBindings.append(location_fw)
    tokenBindings.append(L1_fw)
    tokenBindings.append(L2_fw)

    bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
    bp_in_color_dim = color_act.shape[1]
    bp_in_location_dim = location_act.shape[1]
    bp_in_L1_dim = l1_act.shape[1]
    bp_in_L2_dim = l2_act.shape[1]

    shape_out_all = torch.zeros(bs_testing,
                                bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
    color_out_all = torch.zeros(bs_testing,
                                bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
    location_out_all = torch.zeros(bs_testing,
                                bp_in_location_dim).cuda()  # will be used to accumulate the reconstructed location
    L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
    L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
    BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
    for items in range(bs_testing):  # for each item to be retrieved
        BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
        L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
        L1_out_all[items,:] = L1_out_eachimg / bpPortion  # put the reconstructions into a big tensor and then normalize by the effective # of BP nodes

        L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
        L2_out_all[items, :] = L2_out_eachimg / bpPortion  #

        shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),shape_fw.t()).cuda()  # do the actual reconstruction
        color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
        location_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), location_fw.t()).cuda()
        shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
        color_out_all[items, :] = color_out_eachimg / bpPortion
        location_out_all[items, :] = location_out_eachimg / bpPortion

    return shape_out_all, color_out_all, location_out_all, L2_out_all, L1_out_all

def BPTokens_with_labels(bp_outdim, bpPortion,storeLabels, shape_coef, color_coef, shape_act, color_act,l1_act,l2_act,oneHotShape, oneHotcolor, bs_testing, layernum, normalize_fact ):
    # Store and retrieve multiple items including labels in the binding pool
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items

    with torch.no_grad():  # <---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all = list()  # will be used to accumulate the specific token linkages
        BP_in_all = list()  # will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]
        bp_in_L2_dim = l2_act.shape[1]
        oneHotShape = oneHotShape.cuda()

        oneHotcolor = oneHotcolor.cuda()
        bp_in_Slabels_dim = oneHotShape.shape[1]  # dim =20
        bp_in_Clabels_dim= oneHotcolor.shape[1]


        shape_out_all = torch.zeros(bs_testing,bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out_all = torch.zeros(bs_testing,bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
        L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
        shape_label_out=torch.zeros(bs_testing, bp_in_Slabels_dim).cuda()
        color_label_out = torch.zeros(bs_testing, bp_in_Clabels_dim).cuda()

        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  # make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()
        L2_fw = torch.randn(bp_in_L2_dim, bp_outdim).cuda()
        shape_label_fw=torch.randn(bp_in_Slabels_dim, bp_outdim).cuda()
        color_label_fw = torch.randn(bp_in_Clabels_dim, bp_outdim).cuda()

        # ENCODING!  Store each item in the binding pool
        for items in range(bs_testing):  # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

            if layernum == 1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            elif layernum==2:
                BP_in_eachimg = torch.mm(l2_act[items, :].view(1, -1), L2_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coef + torch.mm(color_act[items, :].view(1, -1), color_fw) * color_coef  # binding pool inputs (forward activations)
                BP_in_Slabels_eachimg=torch.mm(oneHotShape [items, :].view(1, -1), shape_label_fw)
                BP_in_Clabels_eachimg = torch.mm(oneHotcolor[items, :].view(1, -1), color_label_fw)


            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_Slabels_eachimg[:, notLink] = 0
            BP_in_Clabels_eachimg[:, notLink] = 0
            if storeLabels==1:
                BP_in_all.append(
                    BP_in_eachimg + BP_in_Slabels_eachimg + BP_in_Clabels_eachimg)  # appending and stacking images
                notLink_all.append(notLink)

            else:
                BP_in_all.append(BP_in_eachimg )  # appending and stacking images
                notLink_all.append(notLink)



        # now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items, 1)
        BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # divide by the token percent, as a normalizing factor

        BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
        notLink_all = torch.stack(notLink_all)  # this is the set of 0'd connections for each of the tokens

        # NOW REMEMBER
        for items in range(bs_testing):  # for each item to be retrieved
            BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
            if layernum == 1:
                L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
                L1_out_all[items,:] = (L1_out_eachimg / bpPortion ) * normalize_fact # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
            if layernum==2:

                L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
                L2_out_all[items, :] = L2_out_eachimg / bpPortion  #
            else:
                shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),shape_fw.t()).cuda()  # do the actual reconstruction
                color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
                shapelabel_out_each=torch.mm(BP_in_items[items, :].view(1, -1),shape_label_fw.t()).cuda()
                colorlabel_out_each = torch.mm(BP_in_items[items, :].view(1, -1), color_label_fw.t()).cuda()

                shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
                color_out_all[items, :] = color_out_eachimg / bpPortion
                shape_label_out[items,:]=shapelabel_out_each/bpPortion
                color_label_out[items,:]=colorlabel_out_each/bpPortion

    return shape_out_all, color_out_all, L2_out_all, L1_out_all,shape_label_out,color_label_out


def BPTokens_binding_all(bp_outdim,  bpPortion, shape_coef,color_coef,shape_act,color_act,l1_act,bs_testing,layernum, shape_act_grey, color_act_grey):
    #Store multiple items in the binding pool, then try to retrieve the token of item #1 using its shape as a cue
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items
    #layernum= either 1 (reconstructions from l1) or 0 (recons from the bottleneck
    with torch.no_grad(): #<---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all=list()  #will be used to accumulate the specific token linkages
        BP_in_all=list()    #will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottlenecks
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]  # neurons in the Bottleneck
        tokenactivation = torch.zeros(bs_testing)  # used for finding max token
        shape_out = torch.zeros(bs_testing,
                                    bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out= torch.zeros(bs_testing,
                                    bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        l1_out= torch.zeros(bs_testing, bp_in_L1_dim).cuda()


        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  #make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()

        #ENCODING!  Store each item in the binding pool
        for items in range (bs_testing):   # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  #list of 0'd BPs for this token
            if layernum==1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw)+torch.mm(color_act[items, :].view(1, -1), color_fw) # binding pool inputs (forward activations)

            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_all.append(BP_in_eachimg)  # appending and stacking images
            notLink_all.append(notLink)

        #now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items,1)
        BP_in_items = torch.sum(BP_in_items,0).view(1,-1)   #divide by the token percent, as a normalizing factor

        notLink_all=torch.stack(notLink_all)   # this is the set of 0'd connections for each of the tokens

        retrieve_item = 0
        if layernum==1:
            BP_reactivate = torch.mm(l1_act[retrieve_item, :].view(1, -1), L1_fw)
        else:
            BP_reactivate = torch.mm(shape_act_grey[retrieve_item, :].view(1, -1),shape_fw)  # binding pool retreival

        # Multiply the cued version of the BP activity by the stored representations
        BP_reactivate = BP_reactivate  * BP_in_items

        for tokens in range(bs_testing):  # for each token
            BP_reactivate_tok = BP_reactivate.clone()
            BP_reactivate_tok[0,notLink_all[tokens, :]] = 0  # set the BPs to zero for this token retrieval
            # for this demonstration we're assuming that all BP-> token weights are equal to one, so we can just sum the
            # remaining binding pool neurons to get the token activation
            tokenactivation[tokens] = BP_reactivate_tok.sum()

        max, maxtoken =torch.max(tokenactivation,0)   #which token has the most activation

        BP_in_items[0, notLink_all[maxtoken, :]] = 0  #now reconstruct color from that one token
        if layernum==1:

            l1_out = torch.mm(BP_in_items.view(1, -1), L1_fw.t()).cuda() / bpPortion  # do the actual reconstruction
        else:

            shape_out = torch.mm(BP_in_items.view(1, -1), shape_fw.t()).cuda() / bpPortion  # do the actual reconstruction of the BP
            color_out = torch.mm(BP_in_items.view(1, -1), color_fw.t()).cuda() / bpPortion

    return tokenactivation, maxtoken, shape_out,color_out, l1_out


# defining the classifiers
clf_ss = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for shape
clf_sc = svm.SVC(C=10, gamma='scale', kernel='rbf')  #classify shape map against color labels
clf_cc = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for color
clf_cs = svm.SVC(C=10, gamma='scale', kernel='rbf')#classify color map against shape labels


#training the shape map on shape labels and color labels
def classifier_shape_train(whichdecode_use, train_dataset):
    vae.eval()
    with torch.no_grad():
        data, labels  =next(iter(train_dataset))
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()

        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
        z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
        print('training shape bottleneck against color labels sc')
        clf_sc.fit(z_shape.cpu().numpy(), train_colorlabels)

        print('training shape bottleneck against shape labels ss')
        clf_ss.fit(z_shape.cpu().numpy(), train_shapelabels)

#testing the shape classifier (one image at a time)
def classifier_shape_test(whichdecode_use, clf_ss, clf_sc, test_dataset, verbose=0):
    vae.eval()
    with torch.no_grad():
        data, labels  =next(iter(test_dataset))
        test_shapelabels=labels[0].clone()
        test_colorlabels=labels[1].clone()

        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
        z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
        pred_ss = torch.tensor(clf_ss.predict(z_shape.cpu()))
        pred_sc = torch.tensor(clf_sc.predict(z_shape.cpu()))

        SSreport = torch.eq(test_shapelabels.cpu(), pred_ss).sum().float() / len(pred_ss)
        SCreport = torch.eq(test_colorlabels.cpu(), pred_sc).sum().float() / len(pred_sc)

        if verbose ==1:
            print('----*************---------shape classification from shape map')
            print(confusion_matrix(test_shapelabels, pred_ss))
            print(classification_report(test_shapelabels, pred_ss))
            print('----************----------color classification from shape map')
            print(confusion_matrix(test_colorlabels, pred_sc))
            print(classification_report(test_colorlabels, pred_sc))

    return pred_ss, pred_sc, SSreport, SCreport

#training the color map on shape and color labels
def classifier_color_train(whichdecode_use, train_dataset):
    vae.eval()
    with torch.no_grad():
        data, labels  =next(iter(train_dataset))
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()
        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
        z_color = vae.sampling(mu_color, log_var_color).cuda()
        print('training color bottleneck against color labels cc')
        clf_cc.fit(z_color.cpu().numpy(), train_colorlabels)

        print('training color bottleneck against shape labels cs')
        clf_cs.fit(z_color.cpu().numpy(), train_shapelabels)

#testing the color classifier (one image at a time)
def classifier_color_test(whichdecode_use, clf_cc, clf_cs, test_dataset, verbose=0):
    vae.eval()
    with torch.no_grad():
        data, labels  =next(iter(test_dataset))
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()
        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)

        z_color = vae.sampling(mu_color, log_var_color).cuda()
        pred_cc = torch.tensor(clf_cc.predict(z_color.cpu()))
        pred_cs = torch.tensor(clf_cs.predict(z_color.cpu()))

        CCreport = torch.eq(test_colorlabels.cpu(), pred_cc).sum().float() / len(pred_cc)
        CSreport = torch.eq(test_shapelabels.cpu(), pred_cs).sum().float() / len(pred_cs)

        if verbose==1:
            print('----**********-------color classification from color map')
            print(confusion_matrix(test_colorlabels, pred_cc))
            print(classification_report(test_colorlabels, pred_cc))

            print('----**********------shape classification from color map')
            print(confusion_matrix(test_shapelabels, pred_cs))
            print(classification_report(test_shapelabels, pred_cs))

    return pred_cc, pred_cs, CCreport, CSreport



#testing on shape for multiple images stored in memory

def classifier_shapemap_test_imgs(shape, shapelabels, colorlabels,numImg, clf_shapeS, clf_shapeC, test_dataset, verbose = 0):

    global numcolors

    numImg = int(numImg)

    with torch.no_grad():
        predicted_labels=torch.zeros(1,numImg)
        shape = torch.squeeze(shape, dim=1)
        shape = shape.cuda()
        test_colorlabels = thecolorlabels(test_dataset)
        pred_ssimg = torch.tensor(clf_shapeS.predict(shape.cpu()))

        pred_scimg = torch.tensor(clf_shapeC.predict(shape.cpu()))

        SSreport = torch.eq(shapelabels.cpu(), pred_ssimg).sum().float() / len(pred_ssimg)
        SCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_scimg).sum().float() / len(pred_scimg)

        if verbose==1:
            print('----*************---------shape classification from shape map')
            print(confusion_matrix(shapelabels[0:numImg], pred_ssimg))
            print(classification_report(shapelabels[0:numImg], pred_ssimg))
            print('----************----------color classification from shape map')
            print(confusion_matrix(colorlabels[0:numImg], pred_scimg))
            print(classification_report(test_colorlabels[0:numImg], pred_scimg))
    return pred_ssimg, pred_scimg, SSreport, SCreport


#testing on color for multiple images stored in memory
def classifier_colormap_test_imgs(color, shapelabels, colorlabels,numImg, clf_colorC, clf_colorS, test_dataset, verbose = 0):


    numImg = int(numImg)


    with torch.no_grad():

        color = torch.squeeze(color, dim=1)
        color = color.cuda()
        test_colorlabels = thecolorlabels(test_dataset)


        pred_ccimg = torch.tensor(clf_colorC.predict(color.cpu()))
        pred_csimg = torch.tensor(clf_colorS.predict(color.cpu()))


        CCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_ccimg).sum().float() / len(pred_ccimg)
        CSreport = torch.eq(shapelabels.cpu(), pred_csimg).sum().float() / len(pred_csimg)


        if verbose == 1:
            print('----*************---------color classification from color map')
            print(confusion_matrix(test_colorlabels[0:numImg], pred_ccimg))
            print(classification_report(colorlabels[0:numImg], pred_ccimg))
            print('----************----------shape classification from color map')
            print(confusion_matrix(shapelabels[0:numImg], pred_csimg))
            print(classification_report(shapelabels[0:numImg], pred_csimg))

        return pred_ccimg, pred_csimg, CCreport, CSreport

# shape label network
class VAEshapelabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAEshapelabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape


    def sampling_labels (self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)
        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return  z_shape_label

# color label network
class VAEcolorlabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, zlabel_dim):
        super(VAEcolorlabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label = nn.Linear(hlabel_dim, zlabel_dim)  # mu color
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim)  # log-var color

    def sampling_labels(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label = self.fc22label(h)
        z_color_label = self.sampling_labels(mu_shape_label, log_var_shape_label)
        return  z_color_label

# location label network
class VAElocationlabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAElocationlabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape

    def sampling_labels(self, mu, log_var):
        std = torch.exp(0.75 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)
        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label)
        return  z_shape_label
