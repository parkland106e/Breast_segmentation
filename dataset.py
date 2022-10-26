from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pandas as pd
import pydicom
import numpy as np
import torch
import SimpleITK as sitk


class Sub1_3D_dataset(Dataset):
    def __init__(self, directory, csv, transform=False):
        self.directory = directory
        self.csv = csv
        self.data_list = pd.read_csv(self.csv)
        self.transform =transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        id =  self.data_list['ID'][idx]
        input_path = self.directory +'/'+ str(self.data_list['ID'][idx])+'/Resample_SubT1_C1/Resample_SubT1_C1.nii.gz'

        mri_sitk = sitk.ReadImage(input_path)
        mri_img = sitk.GetArrayFromImage(mri_sitk)
        mri_img = mri_img.transpose(2,1,0)

        split_mri_list = [torch.from_numpy(np.reshape(mri_img[:,:,i],((1,)+(512,512)))) for i in range(192)]
        #print(split_mri_list[0].shape,type(split_mri_list[0]))
        '''
        temp = mri_img[:,:,0]

        print(temp.shape)

        temp = np.reshape(temp, ((1,) + temp.shape))
        print(temp.shape)

        print(id)
        print(mri_img.shape)
        '''


        if self.transform:
            split_mri_list = self.transform(split_mri_list)

        return split_mri_list, id