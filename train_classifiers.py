# prerequisites
from classifiers import classifier_shape_train, classifier_color_train, clf_sc, clf_ss, clf_cc, clf_cs, classifier_shape_test, classifier_color_test
from mVAE import load_checkpoint
from joblib import dump, load
from dataset_builder import dataset_builder
from torch.utils.data import DataLoader, ConcatDataset
import torch
import os

folder_path = 'classifier_output' # the output folder for the trained model versions

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

load_checkpoint('output_emnist_recurr/checkpoint_300.pth') # MLR2.0 trained on emnist letters, digits, and fashion mnist

bs = 20000
test_bs = 10000
# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
emnist_dataset, emnist_skip, emnist_test_dataset = dataset_builder('emnist', bs, None, False, None, False, True)
mnist_dataset, mnist_skip, mnist_test_dataset = dataset_builder('mnist', bs, None, False, None, False, True)
#fmnist_dataset, fmnist_skip, fmnist_test_dataset = dataset_builder('fashion_mnist', bs, None, False, None, False, True)

#concat datasets and init dataloaders
train_loader = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_dataset, mnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)
test_loader = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_test_dataset, mnist_test_dataset, mnist_test_dataset]), batch_size=test_bs, shuffle=True,  drop_last= True)

print('training shape classifiers')
classifier_shape_train('cropped', train_loader)
dump(clf_sc, f'{folder_path}/sc.joblib')
dump(clf_ss, f'{folder_path}/ss.joblib')
clf_ss=load('classifier_output/ss.joblib')

pred_ss, pred_sc, SSreport, SCreport = classifier_shape_test('cropped', clf_ss, clf_sc, test_loader)
print('accuracy:')
print('SS:',SSreport)
print('SC:',SCreport)

print('training color classifiers')
classifier_color_train('cropped', train_loader)
dump(clf_cc, f'{folder_path}/cc.joblib')
dump(clf_cs, f'{folder_path}/cs.joblib')

pred_cc, pred_cs, CCreport, CSreport = classifier_color_test('cropped', clf_cc, clf_cs, test_loader)
print('accuracy:')
print('CC:',CCreport)
print('CS:',CSreport)