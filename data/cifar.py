import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from . import data_utils as dutils
from torchvision.transforms import v2
from tqdm import tqdm

class CIFAR5MDataset(Dataset):
    def __init__(self,resnet_type,chunks,ranges,grayscale=False,custom_pre_transforms=None,custom_post_transforms=None,
                 data_dir="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10_diffusion/",
                 exclude_classes=[],
                 for_training=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert len(chunks) == len(ranges)
        self.transform = dutils.ResNet50Transform(resnet_type=resnet_type,grayscale=grayscale,from_pil=False,
                                                  custom_pre_transforms=custom_pre_transforms,
                                                  custom_post_transforms=custom_post_transforms)
        self.data_dir = data_dir
        self.data = []
        self.labels = []
        for c,rg in zip(chunks,ranges):
            with np.load(f"{self.data_dir}/cifar5m_part{c}.npz") as f:
                N = len(f['Y'])
                a = (int(rg[0]*N) if rg[0] <= 1 else rg[0]) if rg[0] is not None else None
                b = (int(rg[1]*N) if rg[1] <= 1 else rg[1]) if rg[1] is not None else None
                selection = slice(a,b)
                self.data.append(torch.tensor(f['X'][selection].transpose(0,3,1,2)))
                self.labels.append(f['Y'][selection])
        self.data = torch.cat(self.data)
        self.labels = np.concatenate(self.labels)
        if len(exclude_classes) > 0:
            if for_training:
                all_classes = sorted(list(set(self.labels)))
                remaining_classes = [lab for lab in all_classes if lab not in exclude_classes]
                reamining_class_labels = np.arange(len(remaining_classes))
                class_map = {c:new for c,new in zip(remaining_classes,reamining_class_labels)}
                def label_changer(x):
                    return class_map[x]
                vfunc = np.vectorize(label_changer)

            mask = np.isin(self.labels, exclude_classes, invert=True)
            self.data = self.data[mask]
            self.labels = self.labels[mask]
            if for_training:
                self.labels = vfunc(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]
    
    def subset(self,a,b):
        return dutils.TransformDataset(self.transform,self.data[a:b],self.labels[a:b])
    
    def random_split(self,fraction):
        N = len(self.data)
        indices = np.arange(N)
        np.random.shuffle(indices)
        split = int(fraction*N)
        return dutils.TransformDataset(self.transform,self.data[indices[:split]],self.labels[indices[:split]]), \
               dutils.TransformDataset(self.transform,self.data[indices[split:]],self.labels[indices[split:]])
    
    def subselection(self,selection):
        return dutils.TransformDataset(self.transform,self.data[torch.tensor(selection)],self.labels[np.array(selection)])