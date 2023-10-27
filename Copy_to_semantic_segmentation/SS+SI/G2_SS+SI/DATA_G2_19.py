import os
import torch
import random
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from os.path import dirname as up
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Pixel-Level class distribution (total sum equals 1.0)
class_distr = torch.Tensor([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.00109, 0.03226, 0.00693])

bands_mean = np.array([0.07155545296812926, 0.07577214664314802, 0.08582589223164594, 0.093932420609203, 
                       0.08551633730567973, 0.07221917997795396, 0.07957506087837904, 0.0665870470001935, 
                       0.06895354901592188, 0.02780747752072814, 0.021035743271076005, -0.2251287804646475, 
                       -0.015639377357381815, 0.041625170543516825, 0.914629217518507, 0.24840744741440698, 
                       -0.027345373610526674, 0.4959900482883121, -0.14796084900530523]).astype('float32')

bands_std = np.array([0.07443211031847291, 0.0748053670913034, 0.07364501103370846, 0.08625745277604097, 
                      0.08430741108128732, 0.08341016594068279, 0.08941792667126941, 0.08220905935731204, 
                      0.08893198070012531, 0.07274526868074464, 0.057384923858361914, 0.16947746854292822, 
                      0.020918917428496078, 0.04629685318481317, 0.07671365059290396, 0.2527602356993238, 
                      0.026856365587477376, 0.33202821388221554, 0.15275395244193313]).astype('float32')


dataset_path = os.path.join(up(up(up(__file__))), 'data')



###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################

class GenDEBRIS(Dataset): # Extend PyTorch's Dataset class
    def __init__(self, mode = 'train', transform=None, standardization=None, path = dataset_path, agg_to_water= True):
        
        if mode=='train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'),dtype='str')
                
        elif mode=='test':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'),dtype='str')
                
        elif mode=='val':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'),dtype='str')
            
        else:
            raise
        
        self.X_PATCHES = []      # Loaded Images
        self.X_INDICES = []      # Loaded Indices
        self.X_TEXTURES = []     # Loaded Textures

        self.y = []           # Loaded Output Masks

        
        for roi in tqdm(self.ROIs, desc = 'Load '+mode+' set to memory'):
            
            roi_folder = '_'.join( ['S2'] + roi.split('_')[:-1] )               # Get Folder Name
            roi_name = '_'.join( ['S2'] + roi.split('_') )                      # Get File Name
            roi_MASK = os.path.join(path, 'patches', roi_folder, roi_name + '_cl.tif')     # Get mask (same mask for all SS,SI,GLCM)
            
            # PATCHES:
            roi_patch_file = os.path.join(path, 'patches', roi_folder, roi_name + '.tif')        # Get file path of patches

            # INDICES:
            roi_indices_file = os.path.join(path, 'indices', roi_folder, roi_name + '_si.tif')        # Get file path of indices

            # TEXTURES:
            roi_textures_file = os.path.join(path, 'texture', roi_folder, roi_name + '_glcm.tif')        # Get file path textures



            ############################## LOAD MASK ##############################

            ds = gdal.Open(roi_MASK)
            temp = np.copy(ds.ReadAsArray().astype(np.int64))
            
            # Aggregation   (md: 1, dens: 2, sps: 3, natm: 4, swater (shallow water): 11 )
            if agg_to_water:
                temp[temp==1]=1      # keep
                temp[temp==2]=2      # keep
                temp[temp==3]=3      # keep      
                temp[temp==4]=4      # keep   
                temp[temp==5]=5      # keep   
                temp[temp==6]=0      
                temp[temp==7]=0
                temp[temp==8]=0
                temp[temp==9]=6      # keep             
                temp[temp==10]=0
                temp[temp==11]=7     # keep            
                temp[temp==12]=8     # keep
                temp[temp==13]=0
                temp[temp==14]=0
                temp[temp==15]=0           

            
            # Categories from 1 to 0
            temp = np.copy(temp - 1)
            ds=None                   # Close file
            
            self.y.append(temp)

            
            ############################## LOAD PATCHES, INDICES, TEXTURES ##############################
            
            # Patches:
            ds_patch = gdal.Open(roi_patch_file)
            temp_patch = np.copy(ds_patch.ReadAsArray())

            ds_patch = None             # Close file
            self.X_PATCHES.append(temp_patch)


            # Indices:
            ds_indices = gdal.Open(roi_indices_file)
            temp_indices = np.copy(ds_indices.ReadAsArray())

            ds_indices = None               # Close file
            self.X_INDICES.append(temp_indices)


            # Textures:
            ds_textures = gdal.Open(roi_textures_file)
            temp_textures = np.copy(ds_textures.ReadAsArray())

            ds_textures = None              # Close file
            self.X_TEXTURES.append(temp_textures)

        
        # Additional attributes
        self.impute_nan = np.tile(bands_mean, (256,256,1))
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.length = len(self.y)
        self.path = path
        self.agg_to_water = agg_to_water



    def __len__(self):
        return self.length
    


    def getnames(self):
        return self.ROIs
    


    def __getitem__(self, index):

        ss = self.X_PATCHES[index]
        si = self.X_INDICES[index]
        glcm = self.X_TEXTURES[index]

        target = self.y[index]


        # CxWxH to WxHxC:       (No change in order for target mask as it is single channel)
        ss = np.moveaxis(ss, [0, 1, 2], [2, 0, 1]).astype('float32')     
        si = np.moveaxis(si, [0, 1, 2], [2, 0, 1]).astype('float32')       
        glcm = np.moveaxis(glcm, [0, 1, 2], [2, 0, 1]).astype('float32')       
        
        stacked_image = np.concatenate((ss,si), axis=2)


        #NaN masks:
        nan_mask = np.isnan(stacked_image)
        stacked_image[nan_mask] = self.impute_nan[nan_mask]
        img = stacked_image

        

        # Transforms:
        if self.transform is not None:
            target = target[:,:,np.newaxis]
            stack = np.concatenate([img, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image
        
            stack = self.transform(stack)

            img = stack[:-1,:,:]
            target = stack[-1,:,:].long()                                    # Recast target values back to int64 or torch long dtype
        
        if self.standardization is not None:
            img = self.standardization(img)


        # Return:
        return img, target



###############################################################
# Transformations                                             #
###############################################################
class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)
    
###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    return 1/torch.log(c + class_distribution)