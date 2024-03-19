from label_network import load_checkpoint_colorlabels, load_checkpoint_shapelabels, s_classes, vae_shape_labels
import torch
from mVAE import vae, load_checkpoint, image_activations, activations
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder
import matplotlib.pyplot as plt
from joblib import dump, load
from torchvision import utils
import torch.nn.functional as F

load_checkpoint('output_emnist_recurr/checkpoint_300.pth')
load_checkpoint_shapelabels('output_label_net/checkpoint_shapelabels5.pth')
load_checkpoint_colorlabels('output_label_net/checkpoint_colorlabels10.pth')
clf_shapeS=load('classifier_output/ss.joblib')

vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# char combinations: [1,3: B], [L,1: U], []

vae.eval()
with torch.no_grad():
    num1 = 21 #1 # first char
    num2 = 1 #3 # second char
    #colors = torch.randint(low=0, high=10, size=(2,))

    # build one hot vectors to be passed to the label networks
    num_labels = F.one_hot(torch.tensor([num1, num2]).cuda(), num_classes=s_classes).float().cuda() # shape
    loc_labels = torch.zeros((2,100)).cuda()
    loc_labels[0][0], loc_labels[1][8] = 1, 1 # location
    #col_labels = F.one_hot(torch.tensor([num1, num2]).cuda(), num_classes=10).float().cuda()

    # generate shape latents from the labels n = noise
    z_shape_labels = vae_shape_labels(num_labels, n = 1)

    # location latent from the location vector
    z_location = vae.location_encoder(loc_labels)

    # pass latents from label network through encoder
    recon_retinal = vae.decoder_retinal(z_shape_labels, 0, z_location, None, 'shape')
    # clamp shape recons to form one image of the combined numbers
    img1 = recon_retinal[0,:,:,6:34]
    img2 = recon_retinal[1,:,:,6:34]
    comb_img = torch.clamp(img1 + img2, 0, 1)
    comb_img = comb_img.view(1,3,28,28)

    
    #comb_img = torch.cat([comb_img, comb_img],0)
    l1,l2,z_shape, z_color, z_location = activations(comb_img)

    pred_ss = clf_shapeS.predict(z_shape.cpu())

    recon_shape = vae.decoder_shape(z_shape, 0, 0)
    utils.save_image(comb_img,'1_3_sim.png')
    utils.save_image(recon_shape,'1_3_sim_recon.png')
    utils.save_image(img1,'img1.png')
    utils.save_image(img2,'img2.png')
    print(pred_ss)
    print(vals[pred_ss[0].item()])