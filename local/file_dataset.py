import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import torch
import h5py


tf = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize(mean=[0.404, 0.404, 0.404], std=[0.154, 0.154, 0.154])])
tf_label = transforms.Compose([
    transforms.ToTensor()
])

class isic_dataset(Dataset):
    def __init__(self, csv_list_path, transform=None):
        self.csv_list = pd.read_csv(csv_list_path, header=None)
      
    def __len__(self):
        return self.csv_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index_image = self.csv_list.iloc[idx][0]
        index_label = self.csv_list.iloc[idx][1]

        image=h5py.File(index_image, mode='r')
        image=image['1']
        image = np.array(image)

        label=h5py.File(index_label, mode='r')
        label=label['1']
        label=np.array(label)
        sample = {'image': tf(image), 'label': tf_label(label)}
        return sample
