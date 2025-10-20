import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
import lightning as pl
from data import data_utils as dutils
from data import toy4vec as toy4vec
from torchvision.transforms import v2
from torchvision.datasets import Imagenette
import numpy as np
from torchvision.datasets import Imagenette, CIFAR10
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from .customImagenette import TensorImagenette
import glob
from .jetclass.dataset import SimpleIterDataset
import pandas as pd
import h5py
import os
import threading
from queue import Queue
import time
import h5py
import copy

class GenericDataModule(pl.LightningDataModule):
    def __init__(self,batch_size=512,num_workers=4,pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {"batch_size":self.batch_size,
                              "num_workers":self.num_workers,
                              "pin_memory":self.pin_memory}
        
class ClassifierDatasetHdf5(GenericDataModule):
    def __init__(self,h5_file,frac_train,frac_val,
                 data_key='embeddings',label_key='labels',
                 seed=145082,**kwargs):
        
        super().__init__(**kwargs)
        
        self.h5_file = h5_file
        self.data_key = data_key
        self.label_key = label_key
        
        with h5py.File(h5_file,'r') as fin:
            self.num_data = fin[data_key].shape[0]
        
        self.num_train = int(self.num_data*frac_train)
        self.num_val = int(self.num_data*frac_val)
        self.num_test = self.num_data - frac_val - frac_train
        self.rng = np.random.default_rng(seed=seed)
        self.perm = self.rng.permutation(self.num_data)

    def classifier_friendly_labels(self,labels):
        distinct = sorted(list(set(labels.astype(int))))
        n_label = len(distinct)
        if distinct != list(range(n_label)):
            new_labels = labels.copy()
            for old,new in zip(distinct,range(n_label)):
                new_labels[labels==old] = new
        else:
            return labels.copy()

    def train_dataloader(self):
        idx = np.sort(self.perm[:self.num_train])
        with h5py.File(self.h5_file,'r') as fin:
            data = fin[self.data_key][idx]
            labels = fin[self.label_key][idx]
        labels_class = self.classifier_friendly_labels(labels)
        dataset = TensorDataset(torch.tensor(data),
                                torch.tensor(labels),
                                torch.tensor(labels_class))
        loader = DataLoader(dataset, shuffle=True, **self.loader_kwargs)
        return loader

    def val_dataloader(self):
        idx = np.sort(self.perm[self.num_train:self.num_train+self.num_val])
        with h5py.File(self.h5_file,'r') as fin:
            data = fin[self.data_key][idx]
            labels = fin[self.label_key][idx]
        labels_class = self.classifier_friendly_labels(labels)
        dataset = TensorDataset(torch.tensor(data),
                                torch.tensor(labels),
                                torch.tensor(labels_class))
        loader = DataLoader(dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        idx = np.sort(self.perm[self.num_train+self.num_val:])
        with h5py.File(self.h5_file,'r') as fin:
            data = fin[self.data_key][idx]
            labels = fin[self.label_key][idx]
        labels_class = self.classifier_friendly_labels(labels)
        dataset = TensorDataset(torch.tensor(data),
                                torch.tensor(labels),
                                torch.tensor(labels_class))
        loader = DataLoader(dataset, shuffle=False, **self.loader_kwargs)
        return loader

class PairwiseSumDataset(GenericDataModule):
    def __init__(self,dim,noise_dim,
                 num_train,num_val,num_test,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.noise_dim = noise_dim
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        self.view_generator = dutils.viewGenerator(dutils.permute_dims(dim),2)

        self.train_data, self.train_labels = self.generate_data(self.num_train)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data, self.train_labels),self.view_generator)

        self.val_data, self.val_labels = self.generate_data(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)

        self.test_data, self.test_labels = self.generate_data(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)


    def generate_data(self,N):
        data = torch.rand(N,self.dim+self.noise_dim)
        sums = dutils.pairwise_product_sum(data[:,:self.dim])
        labels = (sums > 0.25).float().reshape(-1,1)
        return data,labels
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
    
class ImagenetteDataset(GenericDataModule):
    def __init__(self,image_width,sup_simclr=False,**kwargs):
        super().__init__(**kwargs)
        
        if sup_simclr:
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                #v2.RandomResizedCrop(image_width),
                v2.Resize(256),
                v2.CenterCrop(image_width),
                v2.ToDtype(torch.float32,scale=True)
            ])
            self.simclr_views = self.simclr_augment
        else:
            # augmentations from original simCLR paper on ImageNet
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                v2.RandomResizedCrop(image_width),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23)],p=0.5),
                v2.ToDtype(torch.float32,scale=True)
            ])
            # view generator for getting two augmentations per image
            self.simclr_views = dutils.viewGenerator(self.simclr_augment,2)

        # augmentations for ImageNet test evaluation - just resize and crop
        self.test_augment = v2.Compose([v2.PILToTensor(),
                                        v2.Resize(256),
                                        v2.CenterCrop(image_width),
                                        v2.ToDtype(torch.float32,scale=True)
                                        ])
        
        # Imagenette datasets
        self.train_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                           split='train',
                           size='full',
                           download=False,
                           transform=self.simclr_views)
        self.val_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.simclr_views)
        self.test_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.test_augment)
        
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader

class ToyJetDataset(GenericDataModule):
    def __init__(self,npart,num_train,num_val,num_test,nrand=16,
                 **kwargs):
        super().__init__(**kwargs)
        self.npart = npart
        self.nrand = nrand
        self.num_train = num_train
        self.num_val   = num_val
        self.num_test  = num_test
        self.jdgs     = toy4vec.jet_data_generator("signal",npart, npart, True,nrandparticle=nrand)
        self.jdgb     = toy4vec.jet_data_generator("background",npart, npart, True,nrandparticle=nrand)
        self.jdgd     = toy4vec.jet_data_generator("signal_data",npart, npart, True,nrandparticle=nrand)
        
        self.view_generator = dutils.viewGenerator(dutils.smearAndRotate(),2)
        self.train_data, self.train_labels = self.generate_mc(self.num_train)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data, self.train_labels),self.view_generator)
        self.train_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)
        
        self.val_data, self.val_labels = self.generate_mc(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)
        self.val_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.test_data, self.test_labels = self.generate_mc(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.true_data, self.true_labels = self.generate_data(self.num_test)
        self.true_dataset = TensorDataset(self.true_data, self.true_labels)
        self.true_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

        self.trut_data, self.trut_labels = self.generate_data(self.num_test)
        self.trut_dataset = TensorDataset(self.true_data, self.true_labels)
        self.trut_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

    def generate_mc(self,n):
        sig,_,_=self.jdgs.generate_dataset(n)
        bkg,_,_=self.jdgb.generate_dataset(n)
        data   = torch.cat((torch.tensor(sig),torch.tensor(bkg)))
        labels = torch.cat((torch.ones(len(sig)),torch.zeros(len(bkg))))
        return data,labels

    def generate_data(self,n):
        sig,_,_=self.jdgd.generate_dataset(n)
        bkg,_,_=self.jdgb.generate_dataset(n)
        data   = torch.cat((torch.tensor(sig),torch.tensor(bkg)))
        labels = torch.cat((torch.ones(len(sig)),torch.zeros(len(bkg))))
        return data,labels
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader

class FlatDataset(GenericDataModule):
    def __init__(self,nsigs,ndisc,num_train,num_val,num_test,nrand=16,skip=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.nsigs  = nsigs
        self.ndisc  = ndisc
        self.nrand  = nrand
        self.num_train = num_train
        self.num_val   = num_val
        self.num_test  = num_test
        self.rand_matrix = self.random_rotation_matrix(ndisc+nrand)
        if skip < 0: 
            self.skip      = nsigs-1
        else:
            self.skip      = skip
        
        self.mins =[]
        self.maxs =[]
        self.peaks=[]
        self.mins.append(0); self.maxs.append(1); self.peaks.append(0.05)
        self.mins.append(0); self.maxs.append(1); self.peaks.append(1.-0.05)
        for pSig in range(2,self.nsigs):
            pMin  = np.random.uniform(0,0.5)
            pMax  = np.random.uniform(0.5,1.0)
            pPeak = np.random.uniform(pMin,pMax)
            self.mins.append(pMin)
            self.maxs.append(pMax)
            self.peaks.append(pPeak)
        print(" Mins:",self.mins,"\n Maxs:",self.maxs,"\n Peaks:",self.peaks)

        self.view_generator = dutils.viewGenerator(dutils.smear,2)
        self.train_data, self.train_labels = self.generate(self.num_train)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data[self.train_labels != self.skip], self.train_labels[self.train_labels != self.skip]),self.view_generator)
        self.train_dataset_basic = dutils.GenericDataset(self.train_data[self.train_labels != self.skip], self.train_labels[self.train_labels != self.skip])
        self.train_dataset_basic_full = dutils.GenericDataset(self.train_data, self.train_labels)
        
        self.val_data, self.val_labels = self.generate(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)
        self.val_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.test_data, self.test_labels = self.generate(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.true_data, self.true_labels = self.generate(self.num_test,True)
        self.true_dataset = TensorDataset(self.true_data, self.true_labels)
        self.true_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

        self.trut_data, self.trut_labels = self.generate(self.num_test,True)
        self.trut_dataset = TensorDataset(self.true_data, self.true_labels)
        self.trut_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

    def random_rotation_matrix(self,dim):
        # Generate a random orthogonal matrix
        random_matrix = np.random.randn(dim, dim)
        Q, R = np.linalg.qr(random_matrix)
        # Ensure the determinant is 1 to represent a proper rotation
        D = np.diag(np.sign(np.diag(R)))
        return Q @ D

    def generate(self,n,iData=False,iMix=False):
        #Generate a clear signal and background using same variables
        #Add some random signals that use same discriminating variables
        #for now, we just do many different traingle distributions
        ndim = self.ndisc+self.nrand
        data = np.empty((self.nsigs,n,ndim))
        for pVar in range(self.nrand):
            data[:,:,pVar+self.ndisc] = np.random.uniform(0.0,1,(self.nsigs,n))
        shift=0.
        if iData == 1:
            shift=0.1
        for pVar in range(self.ndisc):
            for pSig in range(self.nsigs):
                pShift=shift
                if self.maxs[pSig]-self.peaks[pSig] < shift:
                    pShift = self.maxs[pSig]-self.peaks[pSig]-0.01
                data[pSig,:,pVar]=np.random.triangular(self.mins[pSig],self.peaks[pSig]+pShift,self.maxs[pSig], n)
        if iMix:
            m=self.rand_matrix
            m=np.tile(m, (self.nsigs,n, 1,1))
            dtmp = np.reshape(data,(self.nsigs,n,1,ndim))
            stmp = np.matmul(dtmp , m)
            data[:,:,:] = stmp[:,:,0,:]
        data = data.reshape(self.nsigs*n,ndim)
        labels = np.ones((self.nsigs*n))
        for pArr in range(self.nsigs):
            labels[pArr*n:(pArr+1)*n] *= pArr
        return torch.tensor(data),torch.tensor(labels)


    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
class NoisyImagenetteDataset(GenericDataModule):
    def __init__(self,image_width,eps=0.2,p=0.5,sup_simclr=False,**kwargs):
        super().__init__(**kwargs)
        
        if sup_simclr:
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                v2.Resize(256),
                v2.CenterCrop(image_width),
                v2.ToDtype(torch.float32,scale=True),
                v2.RandomApply([v2.GaussianNoise(eps)],p=p)
            ])
            self.simclr_views = self.simclr_augment
        else:
            # augmentations from original simCLR paper on ImageNet
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                v2.RandomResizedCrop(image_width),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23)],p=0.5),
                v2.ToDtype(torch.float32,scale=True)
            ])
            # view generator for getting two augmentations per image
            self.simclr_views = dutils.viewGenerator(self.simclr_augment,2)

        # augmentations for ImageNet test evaluation - just resize and crop
        self.test_augment = v2.Compose([v2.PILToTensor(),
                                        v2.Resize(256),
                                        v2.CenterCrop(image_width),
                                        v2.ToDtype(torch.float32,scale=True),
                                        v2.RandomApply([v2.GaussianNoise(eps)],p=p)
                                        ])
        
        # Imagenette datasets
        self.train_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                           split='train',
                           size='full',
                           download=False,
                           transform=self.simclr_views)
        self.val_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.simclr_views)
        self.test_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.test_augment)
        
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader

class TensorImagenetteDataset(GenericDataModule):
    def __init__(self,image_width,preload=True,**kwargs):
        super().__init__(**kwargs)
        
        # augmentations from original simCLR paper on ImageNet
        self.simclr_augment = v2.Compose([
            v2.RandomResizedCrop(image_width),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23)],p=0.5),
            v2.ToDtype(torch.float32,scale=True)
        ])
        # view generator for getting two augmentations per image
        self.simclr_views = dutils.viewGenerator(self.simclr_augment,2)

        # augmentations for ImageNet test evaluation - just resize and crop
        self.test_augment = v2.Compose([v2.Resize(256),
                                        v2.CenterCrop(image_width),
                                        v2.ToDtype(torch.float32,scale=True)
                                        ])
        
        # Imagenette datasets
        self.train_dataset = TensorImagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette_tensors/",
                           split='train',
                           size='full',
                           download=False,
                           transform=self.simclr_views,
                           preload=preload)
        self.val_dataset = TensorImagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette_tensors/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.simclr_views,
                                preload=preload)
        self.test_dataset = TensorImagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette_tensors/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.test_augment,
                                preload=preload)
        
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
    
class JetClassDataset(GenericDataModule):
    def __init__(self,classes,input_config,limit_test_files=None,augmenter=None,force_observers=False,
                 balanced_batching=False, samples_per_class=None, views_per_class=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/train_100M/"
        self.val_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/val_5M/"
        self.test_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/test_20M/"
        self.augmenter = augmenter
        self.force_observers = force_observers
        
        self.all_classes = ["qcd","wqq","zqq","ttbar","hbb","hcc","hgg","hww2q1l","hww4q","ttbarlep"]
        self.all_class_labels = {c:i for i,c in enumerate(self.all_classes)}
        self.all_class_fileHeaders = {
            "qcd":"ZJetsToNuNu",
            "wqq":"WToQQ",
            "zqq":"ZToQQ",
            "ttbar":"TTBar",
            "hbb":"HToBB",
            "hcc":"HToCC",
            "hgg":"HToGG",
            "hww2q1l":"HToWW2Q1L",
            "hww4q":"HToWW4Q",
            "ttbarlep":"TTBarLep"
        }

        assert set(classes).issubset(self.all_classes)
        self.classes = classes
        self.input_config = input_config
        self.balanced_batching = balanced_batching
        self.samples_per_class = samples_per_class
        self.views_per_class = views_per_class
        
        self.train_file_dict = {c:glob.glob(f"{self.train_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        self.val_file_dict = {c:glob.glob(f"{self.val_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        self.test_file_dict = {c:glob.glob(f"{self.test_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        if limit_test_files is not None:
            for c in self.classes:
                self.test_file_dict[c] = self.test_file_dict[c][:limit_test_files]

    def train_dataloader(self):
        train_dataset = SimpleIterDataset(
            self.train_file_dict,
            self.input_config,
            for_training=True,
            balanced_batching=self.balanced_batching,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='train',
            async_load=True,
            augmenter=self.augmenter,
            force_observers=self.force_observers
        )
        loader_kwargs = self.loader_kwargs.copy()
        if self.balanced_batching:
            # Set batch_size to match balanced batch size
            loader_kwargs['batch_size'] = len(self.classes) * self.samples_per_class * self.views_per_class
            #loader_kwargs['num_workers'] = 1
        loader = DataLoader(train_dataset,persistent_workers=True,**loader_kwargs)
        return loader
        
    def val_dataloader(self,just_dset=False):
        val_dataset = SimpleIterDataset(
            self.val_file_dict,
            self.input_config,
            for_training=True,
            balanced_batching=self.balanced_batching,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='val',
            async_load=True,
            augmenter=self.augmenter,
            force_observers=self.force_observers
        )
        if just_dset:
            return val_dataset
        loader_kwargs = self.loader_kwargs.copy()
        if self.balanced_batching:
            # Set batch_size to match balanced batch size
            loader_kwargs['batch_size'] = len(self.classes) * self.samples_per_class * self.views_per_class
            #loader_kwargs['num_workers'] = 1
        loader = DataLoader(val_dataset,persistent_workers=True,**loader_kwargs)
        return loader

    def test_dataloader(self):
        test_dataset = SimpleIterDataset(
            self.test_file_dict,
            self.input_config,
            for_training=False,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='test',
            async_load=True,
            augmenter=None,
            force_observers=self.force_observers
        )
        test_kwargs = copy.deepcopy(self.loader_kwargs)
        test_kwargs['num_workers'] = 1
        loader = DataLoader(test_dataset,persistent_workers=True,**test_kwargs)
        return loader
    
class CIFAR10Dataset(GenericDataModule):
    def __init__(self,resnet_type,grayscale=False,custom_pre_transforms=None,custom_post_transforms=None,
                 exclude_classes:list[int]=[],for_training=False,**kwargs):
        super().__init__(**kwargs)
        self.transform = dutils.ResNet50Transform(resnet_type=resnet_type,grayscale=grayscale,from_pil=True,
                                                  custom_pre_transforms=custom_pre_transforms,
                                                  custom_post_transforms=custom_post_transforms)

        self.train_dataset = CIFAR10(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10",
                                    train=True,
                                    download=False,
                                    transform=self.transform)
        self.val_dataset = CIFAR10(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10/",
                                    train=False,
                                    download=False,
                                    transform=self.transform)
        self.test_dataset = CIFAR10(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10/",
                                    train=False,
                                    download=False,
                                    transform=self.transform)
        if len(exclude_classes) > 0:
            print("EXCLUDING CLASSES:",exclude_classes)
            train_mask = np.array([lab not in exclude_classes for lab in self.train_dataset.targets])
            val_mask = np.array([lab not in exclude_classes for lab in self.val_dataset.targets])
            test_mask = np.array([lab not in exclude_classes for lab in self.test_dataset.targets])

            all_classes = sorted(list(set(self.train_dataset.targets)))
            remaining_classes = [lab for lab in all_classes if lab not in exclude_classes]
            remaining_class_labels = np.arange(len(remaining_classes))
            class_map = {c:new for c,new in zip(remaining_classes,remaining_class_labels)}
            def label_changer(x):
                return class_map[x]
            vfunc = np.vectorize(label_changer)
            
            self.train_dataset.targets = np.array(self.train_dataset.targets)[train_mask]
            if for_training:
                self.train_dataset.targets = vfunc(self.train_dataset.targets)
            self.train_dataset.targets = list(self.train_dataset.targets)
            self.train_dataset.data = self.train_dataset.data[train_mask]

            self.val_dataset.targets = np.array(self.val_dataset.targets)[val_mask]
            if for_training:
                self.val_dataset.targets = vfunc(self.val_dataset.targets)
            self.val_dataset.targets = list(self.val_dataset.targets)
            self.val_dataset.data = self.val_dataset.data[val_mask]

            self.test_dataset.targets = np.array(self.test_dataset.targets)[test_mask]
            if for_training:
                self.test_dataset.targets = vfunc(self.test_dataset.targets)
            self.test_dataset.targets = list(self.test_dataset.targets)
            self.test_dataset.data = self.test_dataset.data[test_mask]

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
    
class MultiDomainDataset(GenericDataModule):
    def __init__(self,datasets,domain_labels,**kwargs):
        super().__init__(**kwargs)
        assert len(datasets) == len(domain_labels)
        self.datasets = datasets
        self.domain_labels = domain_labels
        
class CMSOpenData(GenericDataModule):
    def __init__(self,loadMC:bool=True,include_higgs=False,train_frac=0.8,val_frac=0.1,test_frac=0.1,pfn_style=False,use_minor_bkg=False,**kwargs):
        super().__init__(**kwargs)
        self.loadMC = loadMC
        self.include_higgs = include_higgs
        self.data_dir = "/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/opendata_higgs4l/data/"
        self.mc_dir = "/n/holystore01/LABS/iaifi_lab/Users/sambt/datasets/opendata_higgs4l/MC/"
        self.pfn_style = pfn_style
        self.use_minor_bkg = use_minor_bkg
        self.label_map = {
            "higgs": 0,
            "dy": 1,
            "ttbar": 1,
            "zz4e": 2,
            "zz2mu2e": 3,
            "zz4mu": 4
        }

        if loadMC:
            self.data, self.labels = self.load_mc()
        else:
            self.data, self.labels = self.load_data()
        
        perm = torch.randperm(self.data.shape[0])
        self.data = self.data[perm]
        self.labels = self.labels[perm]

        Ntrain = int(len(self.data) * train_frac)
        Nval = int(len(self.data) * val_frac)

        self.train_dataset = TensorDataset(self.data[:Ntrain], self.labels[:Ntrain])
        self.val_dataset = TensorDataset(self.data[Ntrain:Ntrain+Nval], self.labels[Ntrain:Ntrain+Nval])
        self.test_dataset = TensorDataset(self.data[Ntrain+Nval:], self.labels[Ntrain+Nval:])
        
    def load_mc(self):
        ## Drell-Yan
        mc_dy10_11 = pd.read_csv(f'{self.mc_dir}/dy1050_2011.csv',index_col=None, header=0)
        mc_dy50_11 = pd.read_csv(f'{self.mc_dir}/dy50_2011.csv',index_col=None, header=0)
        mc_dy10_12 = pd.read_csv(f'{self.mc_dir}/dy1050_2012.csv',index_col=None, header=0)
        mc_dy50_12 = pd.read_csv(f'{self.mc_dir}/dy50_2012.csv',index_col=None, header=0)
        ## ttbar
        mc_ttbar_11 = pd.read_csv(f'{self.mc_dir}/ttbar2011.csv',index_col=None, header=0)
        mc_ttbar_12 = pd.read_csv(f'{self.mc_dir}/ttbar2012.csv',index_col=None, header=0)
        ##zz
        mc_zz4mu_11 = pd.read_csv(f'{self.mc_dir}/zzto4mu2011.csv',index_col=None, header=0)
        mc_zz2mu2e_11 = pd.read_csv(f'{self.mc_dir}/zzto2mu2e2011.csv',index_col=None, header=0)
        mc_zz4e_11 = pd.read_csv(f'{self.mc_dir}/zzto4e2011.csv',index_col=None, header=0)
        mc_zz4mu_12 = pd.read_csv(f'{self.mc_dir}/zzto4mu2012.csv',index_col=None, header=0)
        mc_zz2mu2e_12 = pd.read_csv(f'{self.mc_dir}/zzto2mu2e2012.csv',index_col=None, header=0)
        mc_zz4e_12 = pd.read_csv(f'{self.mc_dir}/zzto4e2012.csv',index_col=None, header=0)

        # create a combined list of MC
        mc_zz4e = [mc_zz4e_11,mc_zz4e_12]
        mc_zz2mu2e = [mc_zz2mu2e_11, mc_zz2mu2e_12]
        mc_zz4mu = [mc_zz4mu_11, mc_zz4mu_12]
        mc_dy = [mc_dy10_11, mc_dy50_11, mc_dy10_12, mc_dy50_12]
        mc_tt = [mc_ttbar_11, mc_ttbar_12]

        if self.include_higgs:
            mc_higgs_11 = pd.read_csv(f'{self.mc_dir}/higgs2011.csv',index_col=None, header=0)
            mc_higgs_12 = pd.read_csv(f'{self.mc_dir}/higgs2012.csv',index_col=None, header=0)
            mc_higgs = [mc_higgs_11, mc_higgs_12]
            out_mc_sig = pd.concat(mc_higgs,axis=0,ignore_index=True)
            out_mc_sig['label'] = self.label_map['higgs']

        out_mc_bkg_zz4e = pd.concat(mc_zz4e,axis=0,ignore_index=True)
        out_mc_bkg_zz4e['label'] = self.label_map['zz4e']

        out_mc_bkg_zz2mu2e = pd.concat(mc_zz2mu2e,axis=0,ignore_index=True)
        out_mc_bkg_zz2mu2e['label'] = self.label_map['zz2mu2e']

        out_mc_bkg_zz4mu = pd.concat(mc_zz4mu,axis=0,ignore_index=True)
        out_mc_bkg_zz4mu['label'] = self.label_map['zz4mu']

        out_mc_bkg_dy = pd.concat(mc_dy,axis=0,ignore_index=True)
        out_mc_bkg_dy['label'] = self.label_map['dy']

        out_mc_bkg_tt = pd.concat(mc_tt,axis=0,ignore_index=True)
        out_mc_bkg_tt['label'] = self.label_map['ttbar']

        out = pd.concat([out_mc_bkg_zz4e,out_mc_bkg_zz2mu2e,out_mc_bkg_zz4mu],axis=0,ignore_index=True)
        if self.use_minor_bkg:
            out = pd.concat([out,out_mc_bkg_dy,out_mc_bkg_tt],axis=0,ignore_index=True)
        if self.include_higgs:
            out = pd.concat([out,out_mc_sig],axis=0,ignore_index=True)
        
        arrays, labels = self.get_features(out)
        del out
        return arrays, labels
        
    def load_data(self):
        data_year  = [pd.read_csv(f'{self.data_dir}/clean_data_2011.csv',index_col=None, header=0)]
        data_year += [pd.read_csv(f'{self.data_dir}/clean_data_2012.csv',index_col=None, header=0)]
        pdata = pd.concat(data_year,axis=0,ignore_index=True)
        arrays, labels = self.get_features(pdata)
        del pdata
        return arrays, labels
    
    def get_features(self,data):
        data['pt1'] = np.sqrt(data['px1']**2 + data['py1']**2)
        data['pt2'] = np.sqrt(data['px2']**2 + data['py2']**2)
        data['pt3'] = np.sqrt(data['px3']**2 + data['py3']**2)
        data['pt4'] = np.sqrt(data['px4']**2 + data['py4']**2)
        
        features = ['pt1','eta1','phi1','E1','PID1','Q1',
                    'pt2','eta2','phi2','E2','PID2','Q2',
                    'pt3','eta3','phi3','E3','PID3','Q3',
                    'pt4','eta4','phi4','E4','PID4','Q4','label']
        array = data[features].to_numpy()
        array,labels = array[:,:-1],array[:,-1]
        array = torch.tensor(array,dtype=torch.float32)
        labels = torch.tensor(labels,dtype=torch.float32)
        array = (array - array.mean(dim=0))/array.std(dim=0)
        if self.pfn_style:
            array = array.reshape(array.shape[0],4,-1) # 4 leptons per event
        return array, labels
        

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True,**self.loader_kwargs)
        return loader
        
    def val_dataloader(self):
        loader = DataLoader(self.test_dataset,shuffle=True,**self.loader_kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.val_dataset,shuffle=True,**self.loader_kwargs)
        return loader

class GWAKDataset(GenericDataModule):
    def __init__(self,exclude_classes=[],**kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(992)
        self.data_dir = "/n/holystore01/LABS/iaifi_lab/Lab/phlab-neurips25/GWAK/"
        o3_data = np.load(f"{self.data_dir}/O3_dataset_fixed_priors.npz")
        glitch_data = np.load(f"{self.data_dir}/glitches.npz")
        all_data = np.concatenate((o3_data['data'], glitch_data['data']), axis=0)
        all_labels = np.concatenate((o3_data['label'], 9*np.ones(glitch_data['data'].shape[0])), axis=0)

        # exclude classes if specified
        if len(exclude_classes) > 0:
            print("EXCLUDING CLASSES:", exclude_classes)
            mask = np.isin(all_labels, exclude_classes, invert=True)
            all_data = all_data[mask]
            all_labels = all_labels[mask]

        # normalize data along time axis
        all_data = (all_data - all_data.mean(axis=-1,keepdims=True)) / all_data.std(axis=-1,keepdims=True)
        
        # shuffle data
        shuf = self.rng.permutation(all_data.shape[0])
        self.data = torch.tensor(all_data[shuf], dtype=torch.float32)
        self.labels = torch.tensor(all_labels[shuf], dtype=torch.int64)

        # split data
        N = self.data.shape[0]
        Ntrain = int(0.75 * N)
        Ntest = int(0.15 * N)
        self.train_dataset = TensorDataset(self.data[:Ntrain], self.labels[:Ntrain])
        self.test_dataset = TensorDataset(self.data[Ntrain:Ntrain+Ntest], self.labels[Ntrain:Ntrain+Ntest])
        self.val_dataset = TensorDataset(self.data[Ntrain+Ntest:], self.labels[Ntrain+Ntest:])

        # add label names for reference
        self.str_label = {
            1:"SineGaussian",
            2:"BBH",
            3:"Gaussian",
            4:"Cusp",
            5:"Kink",
            6:"KinkKink",
            7:"WhiteNoiseBurst",
            8:"Background",
            9:"Glitches"
        }

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=True, **self.loader_kwargs)
        return loader

class OfflineLIGOData(GenericDataModule):
    def __init__(self, signal_classes, chunk_size=10_000, **kwargs):
        super().__init__(**kwargs)
        self.signal_classes = signal_classes
        self.chunk_size = chunk_size

        self.data_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/LIGO/O4_MDC_background/offline_dataset/"
        self.train_files = [
            self.data_dir + "dataset_HL_SR4096_kernel1.0_3194.h5",
            self.data_dir + "dataset_HL_SR4096_kernel1.0_3698.h5",
            self.data_dir + "dataset_HL_SR4096_kernel1.0_1241.h5"
        ]
        self.val_files = [
            self.data_dir + "dataset_HL_SR4096_kernel1.0_6779.h5"
        ]
        self.test_files = [
            self.data_dir + "dataset_HL_SR4096_kernel1.0_6130.h5"
        ]
        
    def train_dataloader(self):
        dataset = HDF5FullLoader(self.train_files, self.signal_classes, subset_size=self.chunk_size, seed=1683)
        loader = DataLoader(dataset, persistent_workers=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        dataset = HDF5FullLoader(self.val_files, self.signal_classes, subset_size=self.chunk_size, seed=1683)
        loader = DataLoader(dataset, persistent_workers=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        dataset = HDF5FullLoader(self.test_files, self.signal_classes, subset_size=self.chunk_size, seed=1683)
        loader = DataLoader(dataset, persistent_workers=True, **self.loader_kwargs)
        return loader

class HDF5FullLoader(IterableDataset):
    def __init__(self, file_paths, signal_classes, subset_size=10000, seed=23234, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.signal_classes = signal_classes
        self.subset_size = subset_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_per_class_per_file = int(subset_size / (len(self.signal_classes) * len(self.file_paths)))
        
        # Map the full dataset structure
        self.total_size = 0
        
        for f in file_paths:
            with h5py.File(f, "r") as fcurr:
                for isig, sig_class in enumerate(signal_classes):
                    key = f"{sig_class}_data"
                    size = fcurr[key].shape[0]
                    self.total_size += size
        
        # Current batch tracking
        self.current_data = None
        self.current_labels = None
        self.current_index = 0
        
        # Load the first batch
        self._load_next_batch()
    
    def _load_next_batch(self):
        """Load a new batch of data"""
        data_list = []
        labels_list = []
        
        for f in self.file_paths:
            with h5py.File(f, "r") as fcurr:
                for isig, sig_class in enumerate(self.signal_classes):
                    key = f"{sig_class}_data"
                    size = fcurr[key].shape[0]
                    # Randomly select indices for this class
                    indices = np.sort(self.rng.choice(size, self.num_per_class_per_file, replace=False))
                    data_list.append(fcurr[key][indices])
                    labels_list.append(isig * np.ones(self.num_per_class_per_file, dtype=int))
        
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        # Shuffle the loaded subset
        shuf = self.rng.permutation(len(data))
        data = data[shuf]
        labels = labels[shuf]
        
        # Convert to torch tensors
        self.current_data = torch.tensor(data, dtype=torch.float32)
        self.current_labels = torch.tensor(labels, dtype=torch.int32)
        self.current_index = 0
        print(f"Loaded new data batch with {len(self.current_data)} samples")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_data is None or self.current_index >= len(self.current_data):
            # Load the next batch
            self._load_next_batch()
            
        item = self.current_data[self.current_index], self.current_labels[self.current_index]
        self.current_index += 1
        return item
    
    def __len__(self):
        # Return the total size of all data
        return self.total_size

class HDF5ChunkLoader(IterableDataset):
    def __init__(self, file_paths, signal_classes, chunk_size=10000, seed=1683):
        super().__init__()
        self.file_paths = file_paths
        self.signal_classes = signal_classes
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(seed)

        self.indices = []
        self.data_shape = None
        for ifile, file_path in enumerate(file_paths):
            with h5py.File(file_path,"r") as fcurr:
                for isig, sig_class in enumerate(signal_classes):
                    key = f"{sig_class}_data"
                    dataset_size = fcurr[key].shape[0]
                    self.data_shape = fcurr[key].shape[1:]  # Assuming all datasets have the same shape
                    for local_idx in range(dataset_size):
                        self.indices.append([ifile, isig, local_idx])
        self.indices = np.array(self.indices, dtype=int)
        self.num_chunks = len(self.indices) // chunk_size + (1 if len(self.indices) % chunk_size > 0 else 0)

        # Set up first chunk
        self.reset()

    def reset(self):
        shuf = self.rng.permutation(len(self.indices))
        self.indices = self.indices[shuf]

        self.current_chunk = -1
        self.current_chunk_size = 0
        self.current_chunk_index = 0
        self.current_chunk_data = None
        self.current_chunk_labels = None
        self._load_next_chunk()

    def _load_next_chunk(self):
        self.current_chunk += 1
        if self.current_chunk == self.num_chunks:
            self.reset()
        else:
            selected_indices = self.indices[self.current_chunk * self.chunk_size:(self.current_chunk + 1) * self.chunk_size]
            
            self.current_chunk_data = np.zeros((len(selected_indices), *self.data_shape), dtype=np.float32)
            self.current_chunk_labels = np.zeros(len(selected_indices), dtype=np.int32)
            self.current_chunk_size = len(selected_indices)


            selected_files = sorted(list(set(selected_indices[:, 0])))
            for ifile in selected_files:
                with h5py.File(self.file_paths[ifile], "r") as fcurr:
                    selected_classes = sorted(list(set(selected_indices[selected_indices[:, 0] == ifile, 1])))
                    for iclass in selected_classes:
                        key = f"{self.signal_classes[iclass]}_data"
                        
                        to_get = selected_indices[(selected_indices[:, 0] == ifile) & (selected_indices[:, 1] == iclass), 2]
                        placement = np.argwhere((selected_indices[:, 0] == ifile) & (selected_indices[:, 1] == iclass))[:,0]

                        print("placement shape",placement.shape)
                        print("get shape",to_get.shape)
                        
                        srt_indices = np.argsort(to_get)
                        unsrt_indices = np.argsort(srt_indices)
                        
                        data_chunk = fcurr[key][to_get[srt_indices]][unsrt_indices]
                        self.current_chunk_data[placement] = data_chunk
                        self.current_chunk_labels[placement] = iclass

            self.current_chunk_index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.indices)
    
    def __next__(self):
        if self.current_chunk_index >= self.current_chunk_size:
            self._load_next_chunk()
        data = self.current_chunk_data[self.current_chunk_index]
        label = self.current_chunk_labels[self.current_chunk_index]
        self.current_chunk_index += 1
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.int32)
    