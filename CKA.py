'''
Implementation of centered kernel alignment (CKA) according to arXiv:1905.00414
Code compares three models of VGG16 architecture trained with different learning paradigms
Adapt models and files to your needs
'''

import torch
import AE_VGG
import Classifier_VGG
import numpy as np
import copy
import datetime
from PIL import Image
import torchvision.transforms as transforms
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import KMeans
import AE_VGG_DEC
from Modules.Utils import cluster_accuracy, target_distribution
import Cluster

'''
Define transformations to be applied on images before making predictions
'''
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize(128),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])

'''
Get image paths
Reshape list if necessary
'''
with open('Paths_Cropped_Ims.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
im_paths  = np.array(data, dtype = str)
im_paths = im_paths[0]



'''
Class image dataset for loading images 
Create with: 
	File_Name: list of filepaths to images 
	transform: Instance of transforms.Compose (optional)
Methods: 
	getitem --> Returns image for index
		--> If transforms is not None, image is transformed by transforms
	len --> Returns number of image paths in filepaths
'''
class ImageDataset(Dataset):
    def __init__(self, File_Name, transform=False):
        self.image_paths = File_Name
        self.transform = transform
    def __len__(self):
        return(len(self.image_paths))
    def __getitem__(self, idx):
        im_path = self.image_paths[idx]
        im_path = im_path.replace("'", "")
        
        im = Image.open('Cropped_Ims/' + im_path)
        im = np.array(im)
        if(self.transform is not None):
            im = self.transform(im)
        return im

'''
Create instances of ImageDataset for data
Put them in a DataLoader
'''
my_data = ImageDataset(im_paths, transform_test)
my_loader = DataLoader(my_data, batch_size = 8, shuffle = False, num_workers = 0, drop_last = True)


'''
Implementation of Hilbert-Schimidt Independence Criterion (HSIC)
Calculates HSIC for two kernel matrices 
'''
def HSIC(K, L):
    n = K.size(0)
    H = torch.eye(n) - 1.0 / n * torch.ones((n, n))
    H = H.to(K.device)
    KH = torch.mm(K, H)
    LH = torch.mm(L, H)
    HSIC = torch.trace(torch.mm(KH, LH)) / (n - 1) ** 2
    return HSIC

'''
Calculate CKA with linear kernel
'''
def kernel_CKA(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    K = X @ X.t()
    L = Y @ Y.t()
    HSIC_KL = HSIC(K, L)
    norm_K = torch.sqrt(HSIC(K, K))
    norm_L = torch.sqrt(HSIC(L, L))
    return HSIC_KL / (norm_K * norm_L)


'''
Specific for your application 
Adapt models and visualizations
Code below compares three models based on VGG16 architecture

'''
device = torch.device('cpu')
classifier = Classifier_VGG.VGG16_GS(2).to(device)
classifier.load_state_dict(torch.load('Classifier.pth', map_location = device))
autoencoder = AE_VGG.Autoencoder().to(device)
autoencoder.load_state_dict(torch.load('AE.pth'))
DEC_AE  = AE_VGG_DEC.Autoencoder(8).to(device)
DEC = Cluster.DEC(cluster_number=2, 
                        hidden_dimension = 8, 
                        encoder=DEC_AE).to(device)
DEC.load_state_dict(torch.load('DEC.pth', map_location = device))

ckas = np.zeros([13, 3])
add = 0
for ims in iter(my_loader):
    ims2 = copy.deepcopy(ims)
    ims3 = copy.deepcopy(ims)
    ims3 = ims.to(device)
    ims = ims.to(device)
    ims2 = ims2.to(device)
    outs_classifier, pred = classifier.forward(ims)
    outs_autoencoder, reconstructed =autoencoder.forward(ims2)
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, bn, _ = DEC.encoder.Encoder.forward(ims3)
    outs_dec = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13]
    #outs_autoencoder = outs_autoencoder.to('cuda:0')
    #outs_autoencoder = uts_classifier.to('cpu')
    for i in range(len(outs_classifier)):
        val = 0.0
        for j, batch in enumerate(outs_classifier[i]):#j = batch in current layer of classifier
            length = len(batch)
            batch = batch.reshape(length, -1)
            batch_ae = outs_autoencoder[i][j].to(device) #Same batch in every layweer of autoencoder
            batch_ae = batch_ae.reshape(length, -1)
            batch_dec = outs_dec[i][j]
            batch_dec = batch_dec.reshape(length, -1)
            cka_ae_cl = kernel_CKA(batch, batch_ae)
            cka_ae_cl = cka_ae_cl.detach().numpy()
            ckas[i][0] += np.mean(cka_ae_cl)
            cka_dec_cl = kernel_CKA(batch, batch_dec)
            cka_dec_cl = cka_dec_cl.detach().numpy()
            ckas[i][1] += np.mean(cka_dec_cl)
            cka_dec_ae = kernel_CKA(batch_ae, batch_dec)
            cka_dec_ae = cka_dec_ae.detach().numpy()
            ckas[i][2] += np.mean(cka_dec_ae)
    add+=1
            
            
    del ims
    del outs_classifier
    del pred
    del reconstructed
    del outs_autoencoder
copy = copy.deepcopy(ckas)
ckas /= add


'''
Visualize results for every layer in heatmap
Save CKA values and heatmaps 
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
labels = ['Classifier', 'Autoencoder', 'DEC']
for i in range(13):
    hm = np.zeros([3,3])
    hm[0,0] = 1
    hm[1,1] = 1
    hm[2,2] = 1
    hm[0,1] = ckas[i][0]
    hm[1,0] = ckas[i][0]
    hm[0,2] = ckas[i][1]
    hm[2,0] = ckas[i][1]
    hm[1,2] = ckas[i][2]
    hm[2,1] = ckas[i][2]
    sns.heatmap(hm, annot = True)
    plt.title(f'CKA for layer {i+1}')
    ax = plt.gca()
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.savefig(f'CKA for layer {i+1}.png')
    plt.show()
    pd.DataFrame(hm).to_csv(f'CKA_layer_{i+1}.csv')
    
    
    
