from label_network import load_checkpoint_colorlabels, load_checkpoint_shapelabels, s_classes, vae_shape_labels
import torch
from mVAE import vae, load_checkpoint, image_activations, activations
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder
import matplotlib.pyplot as plt
from joblib import dump, load
from torchvision import utils
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load VAE, label network, and classifiers:
v = '' # which model version to use, set to '' for the most recent
load_checkpoint(f'output_emnist_recurr{v}/checkpoint_300.pth') # load VAE
load_checkpoint_shapelabels(f'output_label_net{v}/checkpoint_shapelabels5.pth') # load shape label net
clf_shapeS = load(f'classifier_output{v}/ss.joblib')

vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

vae.eval()

output_folder_path = 'letter_sim_output' # the output folder for the generated simulations

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

def letter_sim(char_1, char_2, l_1, l_2, noise = 1, save_img = False):
    # char_1, char_2: type: integer, which characters to combine, corresponds to the index of the desired character in the vals list ie: 4=4, A=10, Z=35..
    # l_1, l_2: type: integer, the location of the first and second character respectively
    # noise: type: float, scales the noise added into the latent representations, set to 1 by default
    # save_img: type: boolean, whether to save the generated images to the output folder
    with torch.no_grad():
        # build one hot vectors to be passed to the label networks
        num_labels = F.one_hot(torch.tensor([char_1, char_2]).cuda(), num_classes=s_classes).float().cuda() # shape labels for input chars
        loc_labels = torch.zeros((2,100)).cuda()
        loc_labels[0][l_1], loc_labels[1][l_2] = 1, 1 # set locations for char1 and 2

        # generate shape latents from the labels, the noise param scales the added normal dist in the sampling call
        z_shape_labels = vae_shape_labels(num_labels, noise)

        # location latent from the location vector
        z_location = vae.location_encoder(loc_labels)

        # pass latents through decoder
        recon_retinal = vae.decoder_retinal(z_shape_labels, 0, z_location, None, 'shape')
        # clamp shape recons to form one image of the combined numbers
        img1 = recon_retinal[0,:,:,(l_1+6):34]
        img2 = recon_retinal[1,:,:,(l_1+6):34]
        comb_img = torch.clamp(img1 + img2, 0, 0.5) * 1.5
        comb_img = comb_img.view(1,3,28,28)

        l1_junk, l2_junk, z_shape, z_color, z_location = activations(comb_img)

        pred_ss = clf_shapeS.predict(z_shape.cpu())
        out_pred = pred_ss[0].item() # predicted character
        pred_prob = clf_shapeS.predict_proba(z_shape.cpu())
        out_prob = pred_prob[0][out_pred]

        if save_img == True:
            recon_shape = vae.decoder_shape(z_shape, 0, 0)
            utils.save_image(comb_img, f'{output_folder_path}/{char_1}_{char_2}_sim.png')
            utils.save_image(recon_shape, f'{output_folder_path}/{char_1}_{char_2}_sim_recon.png')
            utils.save_image(img1, f'{output_folder_path}/{char_1}.png')
            utils.save_image(img2, f'{output_folder_path}/{char_2}.png')
        
    return out_pred, out_prob # returns integer index of predicted character in the vals list, estimated confidence using predict_proba

# Valerie:

# demo:
# generates a 1 and a 3 at positions 0 and 7 respectively, no noise is added to the latent representations and the generated image is saved
# pred, prob = letter_sim(char_1 = 1, char_2 = 3, l_1 = 0, l_2 = 7, noise = 1, save_img = True)
# print(vals[pred])
# print(prob)
# accuracy can be calculated using:
# true_list = [correct_char] * len(prediction_list)
# accuracy = accuracy_score(true_list, prediction_list)

# accuracy can be plotted with matplotlib:
# plt.plot(accuracy_list)
# plt.show()


# testing this out lolz >.<
# initialize lists to score accuracy and confidence 
accuracy_list = []
confidence_list = []
# define the correct character 
correct_char = ‘A’

# iterate over different location pairs 
for l_1 in range(10):
    for l_2 in range(10): 

        # call letter_sim
        pred, prob = letter_sim(char_1 = 1, char_2 = 3, l_1 = l_1, l_2 = l_2, noise=1, save_img = True)

        # calculate accuracy 
        true_list = [correct_char] * 100
        accuracy = accuracy_score(true_list, [pred] * 100)

        # print prediction and confidence
        print(f"Prediction for location pair ({l_1}, {l_2}): {vals[pred]}")
        print(f"Confidence: {prob}")

        # store accuracy and confidence 
        accuracy_list.append(accuracy)
        confidence_list.append(prob)

# add another loop to test each character 100 times 
for char_num in range (10):
# for in range (100):

        # call letter sim for each character
        pred, prob = letter_sim(char_1=char_num, char_2=char_num, l_1 = 0, l_2 = 7, noise=1, saveimg=True)
        
        # calculate accuracy
        true_list = [correct char] * 100
        accuracy = accuracy_score (true_list, [pred] * 100)
        
        # print prediction and confidence
        print(f"Prediction for location pair ({l_1}, {l_2}): {vals[pred]}")
        print (f"Confidence: {prob}")
        
        # store accuracy and confidence
        accuracy_list.append(accuracy)
        confidence_list.append(prob)
        
# calculate the average accuracy and confidence
average_accuracy = sum(accuracy_list) / len (accuracy_list)
average_confidence = sum(confidence_list) / len(confidence_list)
        
# plot accuracy
plt.plot(accuracy list) 
plt.xlabel("Iteration") 
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Iteration")
plt.show()
       
# plot confidence
plt.plot(confidence_list) 
plt.xlabel("Iteration") 
plt.ylabel("Confidence")
plt.title("Confidence vs. Iteration" )
plt.show()
