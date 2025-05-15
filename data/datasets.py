import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import lightning as pl
from . import data_utils as dutils
from . import toy4vec as toy4vec
from torchvision.transforms import v2
#from torchvision.datasets import Imagenette
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from .customImagenette import TensorImagenette
import glob
import scipy.stats as stats
from .jetclass.dataset import SimpleIterDataset
import matplotlib.pyplot as plt
from models.losses import SupervisedSimCLRLoss
#from models.networks import CustomEfficientNet
from models.networks import MLP
from models.litmodels import SimCLRModel,LossCollector
import corner
import matplotlib.lines as mlines


class GenericDataModule(pl.LightningDataModule):
    def __init__(self,batch_size=512,num_workers=2,pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {"batch_size":self.batch_size,
                              "num_workers":self.num_workers,
                              "pin_memory":self.pin_memory}
    
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
    
    def train_dataloader(self,cut_threshold):
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
        self.triangle = False
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
        #Do Basic
        #self.mins.append(0); self.maxs.append(1); self.peaks.append(0.05)
        #self.mins.append(0); self.maxs.append(1); self.peaks.append(1.-0.05)
        #if not self.triangle:
        #    self.mins[1] = 1.
        #    self.maxs[0] = 0.15
        #    self.maxs[1] = 0.15
        #Do assignment
        if self.triangle:
            self.nvars(self.ndisc,self.nsigs)
        else:
            self.nvars_gaus_t2(self.ndisc,self.nsigs)

        self.view_generator = dutils.viewGenerator(dutils.shift(),2)
        self.train_data, self.train_labels = self.generate(self.num_train)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data, self.train_labels),self.view_generator)
        self.train_dataset_basic = dutils.GenericDataset(self.train_data[self.train_labels != self.skip], self.train_labels[self.train_labels != self.skip])
        self.train_dataset_basic_full = dutils.GenericDataset(self.train_data, self.train_labels)
        
        self.val_data, self.val_labels = self.generate(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)
        self.val_dataset_basic = dutils.GenericDataset(self.val_data[self.val_labels != self.skip], self.val_labels[self.val_labels != self.skip])

        self.test_data, self.test_labels = self.generate(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.true_data, self.true_labels = self.generate(self.num_test,True)
        self.true_dataset = TensorDataset(self.true_data, self.true_labels)
        self.true_dataset_basic = dutils.GenericDataset(self.true_data[self.true_labels != self.skip], self.true_labels[self.true_labels != self.skip])
        self.true_dataset_basic_full = dutils.GenericDataset(self.true_data, self.true_labels)

        self.vtr_data, self.vtr_labels = self.generate(self.num_val,True)
        self.vtr_dataset = dutils.AugmentationDataset(TensorDataset(self.vtr_data, self.vtr_labels),self.view_generator)
        self.vtr_dataset_basic = dutils.GenericDataset(self.vtr_data[self.vtr_labels != self.skip], self.vtr_labels[self.vtr_labels != self.skip])
        
        self.trut_data, self.trut_labels = self.generate(self.num_test,True)
        self.trut_dataset = TensorDataset(self.true_data, self.true_labels)
        self.trut_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

    def nvars_rand(self):
        for pSig in range(2,self.nsigs):
            pMin  = np.random.uniform(0,0.5)
            pMax  = np.random.uniform(0.5,1.0)
            pPeak = np.random.uniform(pMin,pMax)
            self.mins.append(pMin)
            self.maxs.append(pMax)
            self.peaks.append(pPeak)
        print(" Mins:",self.mins,"\n Maxs:",self.maxs,"\n Peaks:",self.peaks)
        
    def nvars(self,iD,iNSigs,iNTries=1000,iSigCut=3., iSigMax=10):
        #print("Max:",pairwise_max(iD,[0,1,0.05],[0,1,0.95]))
        ntries=0
        maxes=[]
        for pSig in range(2,iNSigs):
            pPass  = False
            ntries = 0
            pMin = pMax = pPeak = 0
            while pPass == False:
                pMin  = np.random.uniform(0,0.5)
                pMax  = np.random.uniform(0.5,1.0)
                pPeak = np.random.uniform(pMin,pMax)
                tMax = iSigMax
                for pVal in range(len(self.mins)):
                    testMax =  self.pairwise_max(iD,[pMin,pMax,pPeak],[self.mins[pVal],self.maxs[pVal],self.peaks[pVal]])
                    if  tMax > testMax:
                        tMax = testMax
            
                if iSigMax > tMax > iSigCut or ntries > 999:
                    pPass = True
                    maxes.append(tMax)
                ntries += 1
            if ntries < 1000:
                self.mins.append(pMin)
                self.maxs.append(pMax)
                self.peaks.append(pPeak)
            else:
                print("too many tries, reconfigure",ntries)
        print("Maxes:",maxes)
        print("Mins:",self.mins,"\nMaxs:",self.maxs,"\nPeaks:",self.peaks)
        self.choice = []
        for pVar in range(self.ndisc):
            self.choice.append(np.random.choice(np.arange(self.nsigs),self.nsigs,replace=False))
        print("choice",self.choice)
        
    def nvars_gaus_old(self,iD,iNSigs,iNTries=1000,iSigCut=3., iSigMax=5):
        #print("Max:",pairwise_max(iD,[0,1,0.05],[0,1,0.95]))
        ntries=0
        maxes=[]
        lMax = int(iNSigs *2)
        for pSig in range(2,lMax):#iNSigs):
            pPass  = False
            ntries = 0
            pMean = pSig = 0
            while pPass == False:
                pMean   = np.random.uniform(0.,1.)
                pSigma  = np.random.uniform(0.,0.5)
                tMax = iSigMax
                for pVal in range(len(self.mins)):
                    testMax =  self.pairwise_max(iD,[pMean,pSigma],[self.mins[pVal],self.maxs[pVal]])
                    #print("testMax:",testMax,"vals:",pMean,pSigma,"o:",self.mins[pVal],self.maxs[pVal])
                    if  tMax > testMax:
                        tMax = testMax
                if iSigMax > tMax > iSigCut or ntries > 999:
                    pPass = True
                    maxes.append(tMax)
                ntries += 1
            if ntries < 1000:
                self.mins.append(pMean)
                self.maxs.append(pSigma)
            else:
                print("too many tries, reconfigure",ntries)
        print("Maxes:",maxes)
        print("Means:",self.mins,"\nSigmas:",self.maxs)
        self.choice = []
        for pVar in range(self.ndisc):
            #self.choice.append(np.random.choice(np.arange(self.nsigs),self.nsigs,replace=False))
            self.choice.append(np.random.choice(np.arange(lMax),self.nsigs,replace=False))
        print("choice",self.choice)


    def nvars_gaus_other(self,iD,iNSigs,iNTries=5000,iSigCut=1.2, iSigMax=5.):
        #print("Max:",pairwise_max(iD,[0,1,0.05],[0,1,0.95]))
        ntries=0
        maxes=[]
        for pVar in range(self.ndisc):
            pId = pVar*iNSigs
            for pSig in range(iNSigs):
                #if pVar == 0 and pSig < 2:
                #    continue
                pPass  = False
                ntries = 0
                #pMean = pSigma = 0
                while pPass == False:
                    pMean   = np.random.uniform(0.,1.,iNSigs)
                    pSigma  = np.random.uniform(0.02,0.5,iNSigs)
                    testMax,testMin =  self.pairwise_maxs(iD,pMean,pSigma)
                    if ((iSigMax > testMax > iSigCut) and (iSigMax > testMin > iSigCut)) or ntries > 999:
                        pPass = True
                        maxes.append(testMin)
                    ntries += 1
                if ntries < iNTries:
                    for pId in range(len(pMean)):
                        self.mins.append(pMean[pId])
                        self.maxs.append(pSigma[pId])
                    print("Ntries",ntries)
                else:
                    print("too many tries, reconfigure",ntries)
        print("Maxes:",maxes)
        print("Means:",self.mins,"\nSigmas:",self.maxs)
        self.choice = []
        for pVar in range(self.ndisc):
            #self.choice.append(np.random.choice(np.arange(self.nsigs),self.nsigs,replace=False))
            self.choice.append(pVar*iNSigs+np.random.choice(np.arange(iNSigs),self.nsigs,replace=False))
        print("choice",self.choice)

    def nvars_gaus(self,iD,iNSigs,iNTries=5000,iSigCut=3.5, iSigMax=7.):
        #print("Max:",pairwise_max(iD,[0,1,0.05],[0,1,0.95]))
        ntries=0
        maxes=[]
        for pVar in range(self.ndisc):
            pId = pVar*iNSigs
            for pSig in range(iNSigs):
                #if pVar == 0 and pSig < 2:
                #    continue
                pPass  = False
                ntries = 0
                pMean = pSigma = 0
                while pPass == False:
                    pMean   = np.random.uniform(0.,1.)
                    pSigma  = np.random.uniform(0.02,0.5)
                    tMax = 2*iSigMax
                    if pSig == 0:#refresh
                        tMax = iSigMax-0.01
                    for pVal in range(pId,len(self.mins)):
                        testMax =  self.pairwise_max(iD,[pMean,pSigma],[self.mins[pVal],self.maxs[pVal]])
                        #print("testMax:",testMax,"vals:",pMean,pSigma,"o:",self.mins[pVal],self.maxs[pVal])
                        if  tMax > testMax:
                            tMax = testMax
                    if iSigMax > tMax > iSigCut or ntries > 999:
                        pPass = True
                        maxes.append(tMax)
                    ntries += 1
                if ntries < iNTries:
                    self.mins.append(pMean)
                    self.maxs.append(pSigma)
                else:
                    print("too many tries, reconfigure",ntries)
        print("Maxes:",maxes)
        print("Means:",self.mins,"\nSigmas:",self.maxs)
        self.choice = []
        for pVar in range(self.ndisc):
            #self.choice.append(np.random.choice(np.arange(self.nsigs),self.nsigs,replace=False))
            self.choice.append(pVar*iNSigs+np.random.choice(np.arange(iNSigs),self.nsigs,replace=False))
        print("choice",self.choice)


    def nvars_gaus_t2(self,iD,iNSigs,iNTries=5000,iSigCut=3.5, iSigMax=7.):
        ntries=0
        maxes=[]
        self.mins = np.zeros(self.ndisc*iNSigs)
        self.maxs = np.zeros(self.ndisc*iNSigs)
        for pSig in range(iNSigs):
            if pSig == 0:
                self.mins[0::iNSigs]   = np.random.uniform(0.,1.,self.ndisc)
                self.maxs[0::iNSigs]   = np.random.uniform(0.02,0.5,self.ndisc)
                continue
            for pVar in range(0,self.ndisc):
                pPass  = False
                ntries = 0
                pMean = pSigma = 0
                while pPass == False:
                    pMean   = np.random.uniform(0.,1.)
                    pSigma  = np.random.uniform(0.02,0.5)
                    tMax = 2*iSigMax
                    for pVal in range(pSig):
                        pId = int(pVar*iNSigs+pVal)
                        #print(pId,pVar,iNSigs,pVal,self.mins[4])
                        testMax =  self.pairwise_max(iD,[pMean,pSigma],[self.mins[pId],self.maxs[pId]])
                        if  tMax > testMax:
                            tMax = testMax
                    if iSigMax > tMax > iSigCut or ntries > 999:
                        pPass = True
                        maxes.append(tMax)
                    ntries += 1
                if ntries < iNTries:
                    pId = int(pVar*iNSigs+pSig)
                    self.mins[pId] = pMean
                    self.maxs[pId] = pSigma
                else:
                    print("too many tries, reconfigure",ntries)
        print("Maxes:",maxes)
        print("Means:",self.mins,"\nSigmas:",self.maxs)
        self.choice = []
        for pVar in range(self.ndisc):
            #self.choice.append(np.random.choice(np.arange(self.nsigs),self.nsigs,replace=False))
            self.choice.append(pVar*iNSigs+np.random.choice(np.arange(iNSigs),self.nsigs,replace=False))
        print("choice",self.choice)

    #triangular distribution functions
    def triangular_pdf(self, x, a, b, c):
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=float)

        # Rising edge: a <= x < c
        mask1 = (x >= a) & (x < c)
        pdf[mask1] = 2 * (x[mask1] - a) / ((b - a) * (c - a))

        # Falling edge: c <= x <= b
        mask2 = (x >= c) & (x <= b)
        pdf[mask2] = 2 * (b - x[mask2]) / ((b - a) * (b - c))
    
        return pdf

    def triangular_cdf(self, x, a, b, c):
        x = np.asarray(x)
        cdf = np.zeros_like(x, dtype=float)

        # Case: a < x <= c
        mask1 = (x > a) & (x <= c)
        cdf[mask1] = ((x[mask1] - a) ** 2) / ((b - a) * (c - a))

        # Case: c < x < b
        mask2 = (x > c) & (x < b)
        cdf[mask2] = 1 - ((b - x[mask2]) ** 2) / ((b - a) * (b - c))

        # Case: x >= b
        mask3 = (x >= b)
        cdf[mask3] = 1.0

        return cdf
    
    def triangular_int(self, xmin,xmax,a,b,c):
        lMin=self.triangular_cdf(xmin,a,b,c)
        lMax=self.triangular_cdf(xmax,a,b,c)
        return lMax-lMin

    def gaus_int(self, xmin,xmax,mean,sigma):
        lMin=stats.norm.cdf(xmin,loc=mean,scale=sigma)
        lMax=stats.norm.cdf(xmax,loc=mean,scale=sigma)
        return lMax-lMin

    
    def pairwise_max(self, iD,t1=[],t2=[],iNSig=1e2,iNBkg=1e4):
        if self.triangle:
            xrange=np.linspace(0,1,100)
        else:
            xrange=np.linspace(0.,3*t1[1],300)
        if self.triangle:
            c_val = t1[2]
            ints1=self.triangular_int(c_val-xrange,c_val+xrange,t1[0],t1[1],t1[2])
            ints2=self.triangular_int(c_val-xrange,c_val+xrange,t2[0],t2[1],t2[2])
        else:
            c_val = t1[0]
            ints1=self.gaus_int(c_val-xrange,c_val+xrange,t1[0],t1[1])
            ints2=self.gaus_int(c_val-xrange,c_val+xrange,t2[0],t2[1])
        vals=ints1[1:-1]*iNSig/np.sqrt(ints2[1:-1]*iNBkg+0.1)
        maxval=(np.max(vals[(vals > 0) & (vals < 1e1)]))#**iD
        #return vals**iD
        return maxval

    def pairwise_maxs(self, iD,pMeans=[],pSigmas=[],iNSig=1e2,iNBkg=1e4):
        xrange=np.linspace(0.,3*np.max(pSigmas),300)
        lMaxVal=0
        lMinVal=10
        for pId in range(len(pMeans)):
            c_val = pMeans[pId]
            ints1=self.gaus_int(c_val-xrange,c_val+xrange,pMeans[pId],pSigmas[pId])
            ints2=np.zeros(ints1.shape)
            for pIdS in range(len(pMeans)):
                if pIdS == pId:
                    continue
                ints2+=self.gaus_int(c_val-xrange,c_val+xrange,pMeans[pIdS],pSigmas[pIdS])
            vals=ints1[1:-1]*iNSig/np.sqrt(ints2[1:-1]*iNBkg+0.1)
            maxval=(np.max(vals[(vals > 0) & (vals < 1e1)]))#**iD
            if maxval > lMaxVal:
                lMaxVal=maxval
            if maxval < lMinVal:
                lMinVal=maxval
        #return vals**iD
        return lMaxVal,lMinVal
    
    def random_rotation_matrix(self,dim):
        # Generate a random orthogonal matrix
        random_matrix = np.random.randn(dim, dim)
        Q, R = np.linalg.qr(random_matrix)
        # Ensure the determinant is 1 to represent a proper rotation
        D = np.diag(np.sign(np.diag(R)))
        return Q @ D

    def generate(self,n,iData=False,iMix=True):
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
            #0.1 + 0.01*x
        for pVar in range(self.ndisc):
            for pSig,pAxis in enumerate(self.choice[pVar]):
                pShift=shift
                if self.triangle:
                    if self.maxs[pAxis]-self.peaks[pAxis] < shift:
                        pShift = self.maxs[pAxis]-self.peaks[pAxis]-0.01
                    data[pSig,:,pVar]=np.random.triangular(self.mins[pAxis],self.peaks[pAxis]+pShift,self.maxs[pAxis], n)
                else:
                    data[pSig,:,pVar]=(np.random.normal(loc=self.mins[pAxis],scale=self.maxs[pAxis], size=n))
                    if iData: 
                        def shiftfunc(x):
                            return x*1.01+0.1
                        data[pSig,:,pVar]=shiftfunc(data[pSig,:,pVar])
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
        return torch.tensor(data).float(),torch.tensor(labels).long()


    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                            pin_memory=self.pin_memory, num_workers=self.num_workers,persistent_workers=True)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset_basic, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers,persistent_workers=True)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader

    def prepcut(self,idata,imodel,iLabel=None,cut_threshold=0.5):
        with torch.no_grad():
            i_out = (imodel(idata.float(),embed=True))
            i_out=torch.nn.functional.softmax(imodel.classifier(i_out)).numpy()
        i_maxval=np.max(i_out,1)
        cut_i   = idata[i_maxval > cut_threshold]
        if iLabel is not None:
            cut_l = iLabel[i_maxval > cut_threshold]
        else:
            cut_l = torch.tensor(np.argmax(i_out,axis=1))[i_maxval > cut_threshold]
        cut_ds    = dutils.GenericDataset(cut_i, cut_l)
        return cut_ds
    
    def train_datamcloader(self,imodel,cut_threshold=-1,iContamination=100): # No signal contamination in data
        cut_mc = self.prepcut(self.train_data[self.train_labels != self.skip], imodel,self.train_labels[self.train_labels != self.skip],cut_threshold=cut_threshold)
        cut_ds = self.prepcut(self.true_data [self.true_labels  != self.skip], imodel,                cut_threshold=cut_threshold)
        #cut_ds = self.prepcut(self.test_data [self.test_labels  != self.skip], imodel,                cut_threshold=cut_threshold)
        print("No contamin1")
        if iContamination > 0:
            tmp = self.true_data [self.true_labels  == self.skip][0:iContamination]
            tmp_mrg = torch.cat((self.true_data [self.true_labels  != self.skip],tmp))
            cut_ds =  self.prepcut(tmp_mrg, imodel,cut_threshold=cut_threshold)
        
        merger = dutils.ConcatWithLabels([cut_mc,cut_ds],[0,1])
        labels = merger._labels
        num_classes = 2
        sampler  = dutils.BalancedBatchSampler(labels, self.batch_size, num_classes)
        loader   = DataLoader(merger,batch_sampler=sampler,
                              pin_memory=self.pin_memory,num_workers=self.num_workers,persistent_workers=True)
        
        return loader
    
    def val_datamcloader(self,imodel,cut_threshold=-1):
        cut_mc = self.prepcut(self.val_data[self.val_labels != self.skip], imodel,self.val_labels[self.val_labels != self.skip],cut_threshold=cut_threshold)
        cut_ds = self.prepcut(self.vtr_data[self.vtr_labels != self.skip], imodel,               cut_threshold=cut_threshold)

        merger = dutils.ConcatWithLabels([cut_mc,cut_ds],[0,1])
        labels = merger._labels
        num_classes = 2
        sampler  = dutils.BalancedBatchSampler(labels, self.batch_size, num_classes)
        loader   = DataLoader(merger,batch_sampler=sampler,
                              pin_memory=self.pin_memory,num_workers=self.num_workers,persistent_workers=True)

        return loader
    
    def test_datamcloader(self,imodel,cut_threshold=-1,iContamination=0):
        cut_mc = self.prepcut(self.test_data[self.test_labels != self.skip], imodel,self.train_labels[self.test_labels != self.skip],cut_threshold=cut_threshold)
        cut_ds = self.prepcut(self.trut_data[self.trut_labels != self.skip], imodel,                  cut_threshold=cut_threshold)

        merger = dutils.ConcatWithLabels([cut_mc,cut_ds],[0,1])
        labels = merger._labels
        num_classes = 2
        sampler  = dutils.BalancedBatchSampler(labels, self.batch_size, num_classes)
        loader   = DataLoader(merger,batch_sampler=sampler,
                              pin_memory=self.pin_memory,num_workers=self.num_workers)

        return loader
    
    def plot(self):
        input_dim  = self.train_data.shape[1]
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        for var in range(input_dim):
            if var > 8:
                continue
            bins=np.linspace(-0.5,1.5,20)
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 0].numpy(),density=True,alpha=0.5,bins=bins,label='0')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 1].numpy(),density=True,alpha=0.5,bins=bins,label='1')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 2].numpy(),density=True,alpha=0.5,bins=bins,label='2')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 3].numpy(),density=True,alpha=0.5,bins=bins,label='3')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 4].numpy(),density=True,alpha=0.5,bins=bins,label='4')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 5].numpy(),density=True,alpha=0.5,bins=bins,label='5')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 6].numpy(),density=True,alpha=0.5,bins=bins,label='6')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 7].numpy(),density=True,alpha=0.5,bins=bins,label='7')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 7].numpy(),density=True,alpha=0.5,bins=bins,label='8')
            ax[var//3,var % 3].hist(self.train_data[:,var][self.train_labels == 7].numpy(),density=True,alpha=0.5,bins=bins,label='9')
            ax[var//3,var % 3].set_xlabel("var "+str(var))
            ax[var//3,var % 3].set_xlim(-0.5,1.5)
            ax[var//3,var % 3].legend()

        #fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        #for var in range(input_dim):
        #    if var > 8:
        #        continue
        #    _,bins,_=ax[var//3,var % 3].hist(self.true_data[:,var][self.true_labels == self.skip].numpy(),density=True,alpha=0.5,label='skip')
        #    ax[var//3,var % 3].hist(self.true_data[:,var][self.true_labels == 0].numpy(),density=True,alpha=0.5,bins=bins,label='1')
        #    ax[var//3,var % 3].hist(self.true_data[:,var][self.true_labels == 1].numpy(),density=True,alpha=0.5,bins=bins,label='2')
        #    ax[var//3,var % 3].hist(self.true_data[:,var][self.true_labels == 2].numpy(),density=True,alpha=0.5,bins=bins,label='3')
        #    ax[var//3,var % 3].set_xlabel("var "+str(var))
        #    ax[var//3,var % 3].legend()

    def cornerQuick(self,output,output1,labels,labels1):
        #output=torch.sigmoid(output)
        fig = plt.figure(figsize=(8,8))
        corner.corner(output[labels==0].numpy(),fig=fig,color="C0", label='background')
        corner.corner(output[labels==1].numpy(),fig=fig,color="C1", label='signal 1')
        corner.corner(output[labels==2].numpy(),fig=fig,color="C2", label='signal 2')
        corner.corner(output[labels==3].numpy(),fig=fig,color="C3", label='signal 3')
        corner.corner(output1[labels1==0].numpy(),fig=fig,color="C0", label='data background',linestyle="dashed")
        corner.corner(output1[labels1==1].numpy(),fig=fig,color="C1", label='data signal 1',linestyle="dashed")
        corner.corner(output1[labels1==2].numpy(),fig=fig,color="C2", label='data signal 2',linestyle="dashed")
        corner.corner(output1[labels1==3].numpy(),fig=fig,color="C3", label='data signal 3',linestyle="dashed")
        plt.legend(
            handles=[
                mlines.Line2D([], [], color="C0", label='background'),
                mlines.Line2D([], [], color="C1", label='signal 1'),
                mlines.Line2D([], [], color="C2", label='signal 2'),
                mlines.Line2D([], [], color="C3", label='signal 3'),
            ],bbox_to_anchor=(1, 3),frameon=False, loc="upper right"
            )
        plt.show()

    def zscoreplot(self,mc_out,da_out,mc_lab,da_lab,mc_raw,da_raw,intoys=500,plot=True,iOption=0):
        xy1,zscore1,zscore1e=dutils.z_yield  (mc_out,mc_lab,       mc_out,  mc_lab,  self.skip,ntoys=intoys,iNb=10000,iNr=50000,plot=False,iOption=iOption)
        xy1d,zscore1d,zscore1de=dutils.z_yield(da_out,da_lab,       mc_out,  mc_lab,  self.skip,ntoys=intoys,iNb=10000,iNr=50000,plot=False,iOption=iOption)
        xy2,zscore2,zscore2e=dutils.z_yield  (mc_raw,mc_lab,       mc_raw,  mc_lab,  self.skip,ntoys=intoys,iNb=10000,iNr=50000,plot=False,iOption=iOption)
        xy2d,zscore2d,zscore2de=dutils.z_yield(da_raw,da_lab,       mc_raw,  mc_lab,  self.skip,ntoys=intoys,iNb=10000,iNr=50000,plot=False,iOption=iOption)

        if plot:
            plt.plot(xy1,zscore1,c='red',label="trained")
            plt.plot(xy2,zscore2,c='blue',label="raw")
            plt.plot(xy1d,zscore1d,c='red',linestyle='dashed',label="trained(data)")
            plt.plot(xy2d,zscore2d,c='blue',linestyle='dashed',label="raw(data)")
            plt.xlabel("yield")
            plt.ylabel("z-score")
            plt.legend()
            plt.show()

            plt.plot(xy1,zscore1e,c='red',label="trained")
            plt.plot(xy2,zscore2e,c='blue',label="raw")
            plt.plot(xy1d,zscore1de,c='red',linestyle='dashed',label="trained(data)")
            plt.plot(xy2d,zscore2de,c='blue',linestyle='dashed',label="raw(data)")
            plt.xlabel("yield")
            plt.ylabel("z-score")
            plt.legend()
            plt.show()

            
    #def [128,64,32,16]         
    def trainQuick(self,embed_dim=4,hidden_dims=[128,128,128,32],num_epochs=10,batch_size=1000,plot=True,temp=0.01,iFull=False):
        #now contrastive model
        #embed_dim  = 4 #not making it smaller than input space        input_dim  = self.train_data.shape[1]
        input_dim  = self.train_data.shape[1]
        embedder   = MLP(input_dim=input_dim,hidden_dims=hidden_dims,output_dim=embed_dim)#,output_activation="swish")#,dropout=0.1)#.to(device)
        projector  = MLP(input_dim=embed_dim,hidden_dims=[embed_dim],output_dim=embed_dim)
        classifier = MLP(input_dim=embed_dim,hidden_dims=[16,16],output_dim=(self.nsigs))
        shifter    = None#MLP(input_dim=embed_dim,hidden_dims=[32,32,16],output_dim=embed_dim,activation='relu')
        self.model = SimCLRModel(embedder, projector,classifier=classifier,shifter=shifter,lambda_classifier=0.5,temperature=temp,sup_simclr=True)
        #optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.5e-3)
        # Dataloaders
        trainloader = torch.utils.data.DataLoader(self.train_dataset_basic, batch_size=batch_size, shuffle=True,num_workers=self.num_workers,persistent_workers=True)
        if iFull:
            trainloader = torch.utils.data.DataLoader(self.train_dataset_basic_full, batch_size=batch_size, shuffle=True,num_workers=self.num_workers,persistent_workers=True)
        #dutils.train_generic(num_epochs,trainloader,self.model,criterion,optimizer)
        loss_collector = LossCollector()
        trainer = pl.Trainer(max_epochs=num_epochs,callbacks=[loss_collector])
        trainer.fit(model=self.model, train_dataloaders=trainloader, val_dataloaders=self.val_dataloader())
        print(loss_collector.losses[0],"-",loss_collector.losses[-1])
        if loss_collector.losses[-1] > loss_collector.losses[0]*0.8:
            self.model.apply(self.model.init_weights)
            trainer.fit(model=self.model, train_dataloaders=trainloader, val_dataloaders=self.val_dataloader())
        
        mc_lab =self.test_labels.int()
        da_lab=self.trut_labels.int()
        with torch.no_grad():
            output_train  = (self.model(self.train_data.float(),embed=True))
            mc_out  = (self.model(self.test_data.float(),embed=True))
            da_out  = (self.model(self.trut_data.float(),embed=True))

        if plot:
            self.cornerQuick(mc_out,da_out,mc_lab,da_lab)
            self.zscoreplot(mc_out,da_out,mc_lab,da_lab,self.test_data,self.true_data)

        return self.model,output_train,da_out,mc_out,mc_lab,da_lab 

    def trainQuickDataMC(self,cut_threshold=-1,embed_dim=4,hidden_dims=[128,64,32,16],num_epochs=10,batch_size=1000,plot=True,temp=0.1,imodel=None,iMMD=False):
        #now contrastive model
        if imodel is None:
            input_dim  = self.train_data.shape[1]
            embedder   = MLP(input_dim=input_dim,hidden_dims=hidden_dims,output_dim=embed_dim,output_activation="sigmoid",dropout=0.1)#.to(device)
            projector  = MLP(input_dim=embed_dim,hidden_dims=[embed_dim],output_dim=embed_dim)
            classifier = MLP(input_dim=embed_dim,hidden_dims=[16,16],output_dim=(self.nsigs-1))
            self.model = SimCLRModel(embedder, projector,classifier=classifier,shifter=None,lambda_classifier=0.5,temperature=temp,sup_simclr_datamc=True)
        else:
            imodel.sup_simclr_datamc=True
            imodel.sup_simclr=False
            imodel.MMD=iMMD
            self.model = imodel
        # Dataloaders
        trainer = pl.Trainer(max_epochs=num_epochs)
        trainer.fit(model=self.model, train_dataloaders=self.train_datamcloader(imodel=self.model,cut_threshold=cut_threshold), val_dataloaders=self.val_datamcloader(imodel=self.model,cut_threshold=cut_threshold));
        mc_lab =self.test_labels.int()
        da_lab=self.trut_labels.int()
        with torch.no_grad():
            output_train  = (self.model(self.train_data.float(),embed=True))
            mc_out  = (self.model(self.test_data.float(),embed=True))
            da_out  = (self.model(self.trut_data.float(),embed=True))

        if plot:
            self.cornerQuick(mc_out,da_out,mc_lab,da_lab)
            self.zscoreplot(mc_out,da_out,mc_lab,da_lab,self.test_data,self.trut_data,intoys=100)

        return self.model,output_train,da_out,mc_out,mc_lab,da_lab  

    def trainQuickDisc(self,last_dim=16,num_epochs=10,batch_size=1000,plot=True):
        input_dim  = self.train_data.shape[1]
        labels=self.train_labels
        ntrain=10000
        train_nosig = self.train_data[labels != self.skip]
        train_sig   = self.train_data[labels == self.skip]
        ntrain=np.max([ntrain,len(train_sig)])

        ridx=torch.randperm(len(train_nosig))[:ntrain]
        train_nosig1=train_nosig[ridx]
        train_comb = torch.cat((train_nosig1,train_sig[:ntrain]))
        train_lab  = torch.cat((torch.zeros(ntrain),torch.ones(ntrain)))

        dset=dutils.GenericDataset(train_comb,train_lab.long())
        disc_trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
        disc_model=dutils.train_disc(num_epochs,disc_trainloader,input_dim,last_dim=last_dim,output_dim=1)

        mc_lab=self.test_labels
        da_lab=self.test_labels
        with torch.no_grad():
            mc_out = (disc_model(self.test_data.float()))
            da_out = (disc_model(self.trut_data.float()))

        if plot:
            self.zscoreplot(mc_out,da_out,mc_lab,da_lab,self.test_data,self.trut_data,intoys=50,iOption=2)

        return disc_model,da_out,mc_out,mc_lab,da_lab  

    
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
    def __init__(self,classes,input_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/train_100M/"
        self.val_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/val_5M/"
        self.test_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/test_20M/"
        
        self.all_classes = ["qcd","wqq","zqq","ttbar","hbb"]
        self.all_class_fileHeaders = {
            "qcd":"ZJetsToNuNu",
            "wqq":"WToQQ",
            "zqq":"ZToQQ",
            "ttbar":"TTBar",
            "hbb":"HToBB"
        }

        assert set(classes).issubset(self.all_classes)
        self.classes = classes
        self.input_config = input_config
        
        self.train_file_dict = {c:glob.glob(f"{self.train_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        self.val_file_dict = {c:glob.glob(f"{self.val_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        self.test_file_dict = {c:glob.glob(f"{self.test_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}

    def train_dataloader(self):
        train_dataset = SimpleIterDataset(
            self.train_file_dict,
            self.input_config,
            for_training=True,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='train',
            async_load=True
        )
        loader = DataLoader(train_dataset,persistent_workers=True,**self.loader_kwargs)
        return loader
        
    def val_dataloader(self):
        val_dataset = SimpleIterDataset(
            self.val_file_dict,
            self.input_config,
            for_training=True,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='val',
            async_load=True
        )
        loader = DataLoader(val_dataset,persistent_workers=True,**self.loader_kwargs)
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
            name='val',
            async_load=True
        )
        loader = DataLoader(test_dataset,persistent_workers=True,**self.loader_kwargs)
        return loader
    
class CIFAR10Dataset(GenericDataModule):
    def __init__(self,resnet_type,grayscale=False,custom_pre_transforms=None,custom_post_transforms=None,
                 exclude_classes=[],**kwargs):
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
            train_mask = np.array([lab not in exclude_classes for lab in self.train_dataset.targets])
            val_mask = np.array([lab not in exclude_classes for lab in self.val_dataset.targets])
            test_mask = np.array([lab not in exclude_classes for lab in self.test_dataset.targets])
            
            self.train_dataset.targets = list(np.array(self.train_dataset.targets)[train_mask])
            self.train_dataset.data = self.train_dataset.data[train_mask]

            self.val_dataset.targets = list(np.array(self.val_dataset.targets)[val_mask])
            self.val_dataset.data = self.val_dataset.data[val_mask]

            self.test_dataset.targets = list(np.array(self.test_dataset.targets)[test_mask])
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
        
