import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.networks import MLP
from models.losses import SupervisedSimCLRLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset, Sampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2
import numpy as np
from utils.MAHALANOBISutils import compute_empirical_means,compute_empirical_cov_matrix,mahalanobis_test
from utils.ANALYSISutils import plot_2distribution_new
#import utils.NPLMutils as nplm
from scipy.stats import norm,chi2,kstest
from scipy import interpolate
import lmfit
from collections import defaultdict

class viewGenerator:
    """
        This class is used to generate multiple views of the same data point using `transform`.
        This is useful for SimCLR style training where we want to use multiple views of the same data point.
        Intended to be passed as the `transform` argument to a PyTorch DataLoader (e.g. torchvision datasets).
    """
    def __init__(self,transform,n_views):
        self.transform = transform
        self.n_views = n_views
    
    def __call__(self,x):
        return [self.transform(x) for _ in range(self.n_views)]
    
class AugmentationDataset(Dataset):
    """
        A dataset that generates multiple views of the same data point using `viewGenerator` and a `transform`
    """
    def __init__(self,base_dataset,view_generator):
        super().__init__()
        self.base_dataset = base_dataset
        self.view_generator = view_generator

    def __getitem__(self, index):
        """
        Assuming that dataset is spitting out something of the form `(batch,labels)` and we want to
        generate views of batch. TODO: Generalize this to handle arbitrary data formats.
        """
        data = self.base_dataset[index]
        return self.view_generator(data[0]), *data[1:]

    def __len__(self):
        return len(self.base_dataset)
    
class MultiIter:
    def __init__(self,iterators,fractions):
        self.iterators = iterators
        self.fractions = fractions

    def __next__(self):
        i = np.random.choice(len(self.iterators), p=self.fractions)
        return next(self.iterators[i])

class InterleavedIterableDataset(IterableDataset):
    def __init__(self,datasets,fractions):
        """
        Datasets is a list of datasets to interleave. Fractions is a list of fractions for each dataset.
        The fractions should sum to 1.0.
        """
        assert len(datasets) == len(fractions)
        self.datasets = datasets
        self.fractions = fractions
        self.iters = [iter(d) for d in self.datasets]

    def __iter__(self):
        return MultiIter(self.iters,self.fractions)

    def __len__(self):
        return len(self.base_dataset)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        assert batch_size % num_classes == 0, "Batch size must be divisible by number of classes"
        self.samples_per_class = batch_size // num_classes

        # Group indices by class
        self.class_indices = defaultdict(list)
        if len(labels) < 3:
            tmplabel=torch.cat((labels[0],labels[1]))
        else:
            tmplabel=labels
        #print(tmplabel)
        for idx, label in enumerate(tmplabel):
            self.class_indices[label.item()].append(idx)

        # Make sure each class has enough samples
        for c in range(num_classes):
            if len(self.class_indices[c]) < self.samples_per_class:
                raise ValueError(f"Not enough samples for class {c}")

        # Determine how many batches we can generate
        self.num_batches = min(len(indices) // self.samples_per_class for indices in self.class_indices.values())

    def __iter__(self):
        class_indices_copy = {c: indices.copy() for c, indices in self.class_indices.items()}
        for c in class_indices_copy:
            np.random.shuffle(class_indices_copy[c])

        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                selected = class_indices_copy[c][:self.samples_per_class]
                class_indices_copy[c] = class_indices_copy[c][self.samples_per_class:]
                batch.extend(selected)
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def maxlikelihood(iData, iRef, iRefLabel, sig_idx, iNSig, iNBkg, iNBins=100):
    data_sort  = np.sort(iData.flatten().numpy())
    ntot   = len(data_sort)
    width  = int(ntot/iNBins)
    binsRange=[]
    binsRange.append(1.1)
    pVal=data_sort[-1]
    for pBin in range(iNBins-1):
        #pBinHigh = np.where(pBin == 0, 1.1, data_sort[-(iter)*pBin]
        pBinLow =  data_sort[-(width)*(pBin+1)]
        if pVal == pBinLow:
            continue
        binsRange.append(pBinLow)
        pVal=pBinLow
    binsRange.append(-1)
    bins=np.sort(np.array(binsRange))
    data,bin_edges = np.histogram(iData,bins=bins)
    s,_  = np.histogram(iRef[iRefLabel == sig_idx],bins=bins)
    b,_  = np.histogram(iRef[iRefLabel != sig_idx],bins=bins)
    #log likelihood -(x-mu)^2/sigma^2 - log(2pisigma^2)/2 => Delta s^2/sigma^2  => s^2/B
    bscale=iNBkg/np.sum(b)
    sscale=iNSig/np.sum(s)
    expsig=(s*sscale)**2/(b*bscale)
    realsig=((data-bscale*b))**2/(bscale*b)
    sigarr=[]
    expsigarr=[]
    for pBin in range(len(b)):
        ptotsig = np.sum(realsig[-pBin-1:])
        pexpsig = np.sum(expsig [-pBin-1:])
        if ptotsig > 0:
            pval=chi2.cdf(ptotsig, pBin+1)
            ptotsig =   norm.ppf(pval)
        if pexpsig > 0:
            pcen=chi2.ppf(0.5,pBin+1)
            pval=chi2.cdf(pexpsig+pcen, pBin+1)
            print(":",pexpsig)
            pexpsig =   norm.ppf(pval)
            print(pexpsig,pcen,"!",pval)
        sigarr.append(ptotsig)
        expsigarr.append(pexpsig)
    print("sigarr",sigarr,"exp",expsigarr)
    print("val",np.max(sigarr),np.max(expsigarr), sigarr[np.argmax(expsigarr)])
    a
    return sigarr[np.argmax(expsigarr)]
    #low stats version
    #sig=np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    
def ksscore(iData, iRef, iRefLabel, sig_idx):
    ks=kstest(iData.flatten().numpy(), iRef[iRefLabel != sig_idx].flatten().numpy())
    return -1.*norm.ppf(ks.pvalue)
    
### Stuff for pairwise product sum toy dataset ###
def pairwise_product_sum(x,normalize=True):
    if len(x.size()) == 2:
        x = x.unsqueeze(-1) # B, D, 1
    else:
        assert len(x.size()) == 3
    b,n,d = x.shape
    t1 = (x.sum(dim=1)**2).sum(dim=1) # sum(x_i)^2 
    t2 = (x**2).sum(dim=2).sum(dim=1) # sum(x_i^2)
    sums = 0.5 * (t1 - t2) # sum of all pairwise dot products
    if normalize:
        norm_dot = d
        norm_pairwise = 0.5 * (n**2 - n)
        sums = sums / (norm_dot * norm_pairwise)
    return sums

class permute_dims:
    """
        Randomly permute the first `dim` dimensions of x.
    """
    def __init__(self,dim):
        self.dim = dim
    
    def __call__(self,x):
        # assume x is a tensor of shape D where D is the full dimensionality
        randperm = torch.argsort(torch.rand(self.dim))
        aug = x.clone()
        aug[:self.dim] = aug[randperm]
        return aug


class rotate:
    """
        Randomly permute the first `dim` dimensions of x.
    """
    def __call__(self,x):
        # assume x is a tensor of shape D where D is the full dimensionality
        rot = 2.*3.141592741012*torch.rand(1)#x.shape[0]//3)
        x=x.reshape(4,3)
        aug  = self.rotateTheta(x,rot)
        return aug

    def theta_to_eta(self,theta):
        pTheta =  torch.where(theta > np.pi,2*np.pi - theta, theta)            
        return -np.log(np.tan(pTheta/2))

    def eta_to_theta(self,eta):
        return 2 * torch.atan(torch.exp(-eta))

    def convert_to_cart(self, vec):
        px = vec[:,0] * torch.cos(vec[:,2])
        py = vec[:,0] * torch.sin(vec[:,2])
        pz = vec[:,0] * torch.sinh(vec[:,1])
        return torch.stack((px,py,pz)).T

    def convert_to_phys(self,vec):
        pt = torch.sqrt(vec[:, 0]**2 + vec[:, 1]**2)
        phi = torch.atan2(vec[:, 1], vec[:, 0])    
        # Avoid division by zero
        eta = torch.where(pt != 0, torch.asinh(vec[:, 2] / pt), torch.sign(vec[:, 2]) * float('inf'))
        return torch.stack((pt,eta,phi)).T

    def rotateTheta(self,idau,itheta):
        v1  =self.convert_to_cart(idau)
        axis=torch.tensor([1.,0.,0.])
        axisr=axis.repeat((v1.shape[0])).reshape(v1.shape)
        rotmat=self.rotation_matrix_3d(axisr,itheta).float()
        v1  = v1.unsqueeze(2).float()
        #v1  = v1.reshape(v1.shape[0],v1.shape[1],1).float()
        v1rot=torch.matmul(rotmat, v1)
        v1rot=v1rot.squeeze(2)
        #v1rot=v1.reshape(v1.shape[0],v1.shape[1])
        v1rot=self.convert_to_phys(v1rot)
        return v1rot

    def rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        #axis = np.asarray(axis)
        axis = axis / torch.sqrt(torch.dot(torch.tensor(axis),torch.tensor(axis)))
        a = torch.cos(theta / 2.0)
        print("a:",a)
        b, c, d = -axis * torch.sin(theta / 2.0)
        print("b:",b,"c:",c,"d:",d)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return torch.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



    def rotation_matrix_3d(self, axis, angle):
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(float(angle))
    
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        # Ensure correct dimensions for broadcasting
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        if axis.ndim == 1: #single rotation
            rot_matrix = torch.tensor([
                [cos_a + x**2 * (1 - cos_a), x * y * (1 - cos_a) - z * sin_a, x * z * (1 - cos_a) + y * sin_a],
                [y * x * (1 - cos_a) + z * sin_a, cos_a + y**2 * (1 - cos_a), y * z * (1 - cos_a) - x * sin_a],
                [z * x * (1 - cos_a) - y * sin_a, z * y * (1 - cos_a) + x * sin_a, cos_a + z**2 * (1 - cos_a)]
            ])
    
        elif axis.ndim == 2: #batched rotation
            rot_matrix = torch.stack([
                torch.stack([cos_a + x**2 * (1 - cos_a), x * y * (1 - cos_a) - z * sin_a, x * z * (1 - cos_a) + y * sin_a], dim=-1),
                torch.stack([y * x * (1 - cos_a) + z * sin_a, cos_a + y**2 * (1 - cos_a), y * z * (1 - cos_a) - x * sin_a], dim=-1),
                torch.stack([z * x * (1 - cos_a) - y * sin_a, z * y * (1 - cos_a) + x * sin_a, cos_a + z**2 * (1 - cos_a)], dim=-1)
            ], dim=-2)
        
        else:
            raise ValueError("Axis must be a 1D or 2D tensor")
        return rot_matrix


class smear:
    def __call__(self,x):
        # assume x is a tensor of shape D where D is the full dimensionality
        aug=x.reshape(4,3)
        aug[:,1] += (torch.randn(4)*0.1/aug[:,0])
        aug[:,2] += (torch.randn(4)*0.1/aug[:,0])
        return aug.flatten().float()

class shift:
    def __call__(self,x):
        shift=torch.randn(1)*0.1*torch.ones(x.shape)
        shift = x+shift
        return shift.float()
        # assume x is a tensor of shape D where D is the full dimensionality
        #aug=x.copy()
        #print(x.shape)
        #aug += (torch.randn(x.shape)*0.1)
        #aug[:,2] += (torch.randn(4)*0.1/aug[:,0])
        #return x#aug.flatten().float()

    
class smearAndRotate:
#    def __init__(self):
#        self.sme = smear()
#        self.rot = rotate()

    def __call__(self,x):
        #return smear()(x)
        return rotate()(x)
        if torch.rand(1) > 0.5:
            #t = smear()
            return smear()(x)
        else:
            #t = rotate()
            return rotate()(x)
        #return t(x).flatten().float()
    
class GenericDataset(Dataset):
    def __init__(self,data,labels,normalize=False):
        self.data   = data
        self.labels = labels
    
    def generate_augmentation(self,batch):
        return None
    
    def normalize(self,batch):
        return (batch - self.mean.to(batch)) / self.std.to(batch)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        return self.data[idx], self.labels[idx]

def train_generic(inepochs,itrainloader,imodel,icriterion,ioptimizer):
    losses = []
    for epoch in tqdm(range(inepochs)):
        imodel.train()
        epoch_loss = []
        for batch_data, labels in itrainloader:
            batch_data = batch_data.float()

            # Potential to add any augmentation here
            features = imodel(batch_data).unsqueeze(1)
        
            # Compute SimCLR loss
            loss = icriterion(features,labels=labels)
        
            # Backward pass and optimization
            ioptimizer.zero_grad()
            loss.backward()
            ioptimizer.step()
        
            epoch_loss.append(loss.item())
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        #if epoch % 1 == 0:
        #print(f'Epoch [{epoch+1}/{inepochs}], Loss: {mean_loss:.4f}')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)

def train_disc(inepochs,itrain,input_dim,last_dim=16,output_dim=3):
    num_epochs=inepochs
    hidden_dims= [64,128,32,last_dim]
    disc_criterion = nn.CrossEntropyLoss()
    disc_model     = MLP(input_dim=input_dim,hidden_dims=hidden_dims,output_dim=output_dim,output_activation="sigmoid",dropout=0.)#.to(device)
    disc_optimizer = torch.optim.AdamW(disc_model.parameters(), lr=0.5e-2)
    losses = []
    for epoch in tqdm(range(num_epochs)):
        disc_model.train()
        epoch_loss = []
        for batch_data, labels in itrain:
            batch_data = batch_data.float()
            features = disc_model(batch_data)
            if output_dim == 1:
                features=features.squeeze(1)
                loss = disc_criterion(features,labels.float())
            else:
                loss = disc_criterion(features,labels.long())
            disc_optimizer.zero_grad()
            loss.backward()
            disc_optimizer.step()        
            epoch_loss.append(loss.item())
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
    #if epoch % 10 == 0:
    #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss:.4f}')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)
    plt.yscale('log')
    return disc_model


def train_aug(inepochs,itrainloader,imodel,icriterion,ioptimizer):
    losses = []
    for epoch in tqdm(range(inepochs)):
        #imodel.train()
        epoch_loss = []
        for batch_data, labels in itrainloader:
            # Potential to add any augmentation here
            feat0     = imodel(torch.flatten(batch_data[0],start_dim=1)).unsqueeze(1)
            feat1     = imodel(torch.flatten(batch_data[1],start_dim=1)).unsqueeze(1)
            feat      = torch.cat((feat0,feat1),axis=1)
            loss = icriterion(feat)
            # Backward pass and optimization
            ioptimizer.zero_grad()
            loss.backward()
            ioptimizer.step()
            epoch_loss.append(loss.item())
            #print(epoch_loss[-1])
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)
    #plt.yscale('log')
        
#DE-SC0021943 #ECA
#DE-SC001193 #Extra

from torchmetrics import Accuracy,AUROC

def check_disc(itest_data,itest_labels,imodel):
    test_accuracy = Accuracy(task="binary", num_classes=2)#,top_k=2)
    labels=itest_labels.int()
    with torch.no_grad():
        output = (imodel(itest_data.float()))
    print(output.shape[1])
    if output.shape[1] > 1:
        vars = output[:,0]/(output[:,0]+output[:,1])
        print("Accuracy:",test_accuracy(vars[labels < 2],labels[labels < 2]))
    else:
        vars = output.flatten()
        print(vars.shape)
        print("Accuracy:",test_accuracy(vars[labels!=1],labels[labels!=1]//2))

    #plt.plot(output[labels==0][:,0],output[labels==0][:,1],'.',alpha=0.5)
    #plt.plot(output[labels==1][:,0],output[labels==1][:,1],'.',alpha=0.5)
    #plt.plot(output[labels==2][:,0],output[labels==2][:,1],'.',alpha=0.5)    
    plt.hist(vars[labels==0],alpha=0.5)
    plt.hist(vars[labels==1],alpha=0.5)
    #plt.hist(vars[labels==2],alpha=0.5)
    plt.show()

def approxDist(iData, iModel, iLabel, nsamps):
        dists=[]
        for pVal in iData:
                pDist=[]
                for pSamp in range(nsamps):
                    pModel = iModel[iLabel == pSamp]
                    ppDist=torch.sqrt(torch.sum((pModel-pVal)**2,axis=1))
                    pDist.append(ppDist.mean()/ppDist.std())
                dists.append(torch.min(torch.tensor(pDist)))
        return torch.tensor(dists)

from torchmetrics.classification import ROC
def approxAUC(dist, labels,nsamps):
    metric = AUROC(task="binary")
    auc_score = metric(dist, labels//(nsamps-1))
    #test_accuracy = Accuracy(task="binary", num_classes=2)
    print("AUC:",auc_score)
    #print("Acc:",test_accuracy(dist., labels//(nsamps-1)))
    roc = ROC(task="binary")
    roc.update(dist, labels//(nsamps-1))
    fpr, tpr, thresholds = roc.compute()
    plt.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def ResNet50Transform(resnet_type,grayscale=False,from_pil=True,custom_pre_transforms=None,custom_post_transforms=None):
        assert resnet_type in ['resnet18','resnet50']
        if resnet_type == 'resnet50':
            resize_size = 232
            crop_size = 224
        elif resnet_type == 'resnet18':
            resize_size = 256
            crop_size = 224
        else:
            print("resnet type not recognized, using resnet18 values")
            resize_size = 232
            crop_size = 224
        
        transforms = []

        if from_pil:
            transforms.append(v2.PILToTensor()) # CIFAR10 is stored as PIL; cifar5m is not
        
        if custom_pre_transforms is not None: # transforms applied to smaller cifar images before resizing/interpolation
            assert type(custom_pre_transforms) == list
            for t in custom_pre_transforms:
                transforms.append(t)
            
        if grayscale:
            transforms.append(v2.Grayscale(num_output_channels=3))
        
        transforms.append(v2.Resize(resize_size,interpolation=InterpolationMode.BILINEAR,antialias=True)) # standard resnet preprocessing
        transforms.append(v2.CenterCrop(crop_size)) # standard resnet preprocessing

        if custom_post_transforms is not None:
            assert type(custom_post_transforms) == list
            for t in custom_post_transforms:
                transforms.append(t)

        transforms.append(v2.ToDtype(torch.float32,scale=True)) # standard resnet preprocessing
        transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])) # standard resnet normalization

        return v2.Compose(transforms)

class TransformDataset(Dataset):
    def __init__(self,transform,data,labels):
        super().__init__()
        self.transform = transform
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
    def subset(self,a,b):
        return TransformDataset(self.transform,self.data[slice(a,b)],self.labels[slice(a,b)])
    
    def random_split(self,fraction):
        N = len(self.data)
        indices = np.arange(N)
        np.random.shuffle(indices)
        split = int(fraction*N)
        i1, i2 = indices[:split], indices[split:]
        return TransformDataset(self.transform,self.data[i1],self.labels[i1]), \
               TransformDataset(self.transform,self.data[i2],self.labels[i2])
    
    def subselection(self,selection):
        return TransformDataset(self.transform,self.data[torch.tensor(selection)],self.labels[np.array(selection)])
    
class ConcatWithLabels(Dataset):
    def __init__(self, datasets,labels):
        assert len(datasets) == len(labels)
        self._datasets = datasets
        self._labels = [labels[i]*torch.ones(len(datasets[i])) for i in range(len(datasets))]
        self._len = sum(len(dataset) for dataset in datasets)
        self._indexes = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

    def __getitem__(self, index):
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start], self._labels[dataset_index][index - start]

    def __len__(self) -> int:
        return self._len

#def approxDist(iData, iModel, iLabel, nsamps):


def mahalanobis_dist(data, ref, ref_label,plot=True,fit=False,rule='sum'):#, sig_label=-1, seed=0, n_ref=1e4, n_bkg=1e3, n_sig=1e2, z_ratio=0.1, anomaly_type ='', plot=True, pois_ON=False):
    '''
    - computes the mahalnobis test for the dataset 
    '''
    # random seed                                                                                                                    
    #np.random.seed(seed)
    #print('Random seed: '+str(seed))
    
    # train on GPU?                                                                                                                  
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    #data   = data.to(DEVICE)
    #model  = model.to(DEVICE)
    #label  = label.to(DEVICE)

    # estimate parameters of the bkg model 
    means=compute_empirical_means(ref,ref_label)
    emp_cov=compute_empirical_cov_matrix(ref, ref_label, means)
    M_data = mahalanobis_test(data, means, emp_cov)
    if plot:
        M_ref  = mahalanobis_test(ref, means, emp_cov)
        # visualize mahalanobis
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        rMin=torch.min(M_ref)
        rMax=torch.max(M_ref)
        bins=np.linspace(rMin,rMax,20)
        plt.hist(M_ref,bins=bins,label='ref',alpha=0.5)
        plt.hist(M_data,bins=bins,label='data',alpha=0.5)
        #plt.hist([M_ref, M_data], density=True, label=['REF', 'DATA'])
        #font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, ncol=2, loc='best')
        #plt.yscale('log')
        #plt.yticks(fontsize=16, fontname='serif')
        #plt.xticks(fontsize=16, fontname='serif')
        plt.ylabel("density")#, fontsize=22, fontname='serif')
        plt.xlabel("mahalanobis metric")#, fontsize=22, fontname='serif')
        #plt.savefig(output_folder+'distribution.pdf')
        plt.show()
    if fit:
        M_ref  = mahalanobis_test(ref, means, emp_cov)
        result=fitDiff(-1.*M_data,-1.*M_ref)

    if rule=='sum':
        t = -1* torch.sum(M_data)
    elif rule=='max':
        t = -1* torch.min(M_data)
    #print('Mahalanobis test: ', "%f"%(t))
    return t,-1.*M_data

def gausSpline(x,mean,sigma,a1,a2,iTck=None):
    sig = norm.pdf(x, mean, sigma)*a1
    bkg = interpolate.splev(x, iTck)*a2
    #print(mean,sigma,a1,a2)
    #print("bkg:",bkg)
    return sig+bkg

def spline(x,a2,iTck=None):
    bkg = interpolate.splev(x, iTck)*a2
    return bkg

def fitDiff(data,ref):
    #start with binned fit to be easy
    rMin=torch.min(ref)
    rMax=torch.max(ref)
    bins=np.linspace(rMin,rMax,20)
    refhist,bin_edges  = np.histogram(ref, bins=bins)
    datahist,_         = np.histogram(data, bins=bins)
    x                  = 0.5*(bin_edges[1:] + bin_edges[:-1])
    tck                = interpolate.splrep(x, refhist)
    smodel             = lmfit.Model(gausSpline)
    bmodel             = lmfit.Model(spline)
    ps = smodel.make_params(mean=1,sigma=0.2,a1=100.0,a2=1.0)
    pb = bmodel.make_params(a2=1.)
    weights = 1./np.sqrt(np.maximum(refhist,0.1))
    resultb = bmodel.fit(data=datahist,params=pb,x=x,weights=weights,iTck=tck)
    lmfit.report_fit(resultb)
    results = smodel.fit(data=datahist,params=ps,x=x,weights=weights,iTck=tck)
    lmfit.report_fit(results)
    #plt.errorbar(x,datahist,yerr=np.sqrt(datahist),marker='o')
    #plt.errorbar(x,refhist*resultb.params['a2'].value,yerr=np.sqrt(refhist),marker='o')
    #plt.yscale('log')
    #plt.show()
    #return resultb
    #results.plot()
    #return resultsb.chisq-results.chisq
    return results

def zscore(t1,t2,iPrint=False):
    df=np.median(t2)
    Z1_obs     = -norm.ppf(chi2.sf(np.median(t1), df))
    t1_obs_err = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    Z1_obs_p   = -norm.ppf(chi2.sf(np.median(t1)+t1_obs_err, df))
    Z1_obs_m   = -norm.ppf(chi2.sf(np.median(t1)-t1_obs_err, df))
    if iPrint:
        print("z1",Z1_obs,"+",Z1_obs_p,"-",Z1_obs_m)
    
    Z2_obs     = -norm.ppf(chi2.sf(np.median(t2), df))
    t2_obs_err = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z2_obs_p   = -norm.ppf(chi2.sf(np.median(t2)+t2_obs_err, df))
    Z2_obs_m   = -norm.ppf(chi2.sf(np.median(t2)-t2_obs_err, df))
    if iPrint:
        print("z2",Z2_obs,"+",Z2_obs_p,"-",Z2_obs_m)
    return Z1_obs

def zemp(t1,t2,iPrint=False):
    t_empirical = np.sum(1.*(t2>np.mean(t1)))*1./t2.shape[0]
    empirical_lim = '='
    if t_empirical==0:
        empirical_lim='>'
        t_empirical = 1./t1.shape[0]
    t_empirical_err = t_empirical*np.sqrt(1./np.sum(1.*(t2>np.mean(t1))+1./t1.shape[0]))
    Z_empirical = norm.ppf(1-t_empirical)
    Z_empirical_m = norm.ppf(1-(t_empirical+t_empirical_err))
    Z_empirical_p = norm.ppf(1-(t_empirical-t_empirical_err))
    if iPrint:
        print("zemp",Z_empirical,"+",Z_empirical_p,"-",Z_empirical_m,t_empirical,t_empirical_err)
    return Z_empirical
    
def run_toy( nsig, nbkg, nref, data, labels, model, model_labels,sig_idx,ntoys=1000,plot=True,iOption=0):
    t_sig = []
    t_ref = []
    refs      = model       [model_labels != sig_idx]
    refs_label= model_labels[model_labels != sig_idx]
    srefs    = model       [model_labels == sig_idx]
    sigs     = data[labels == sig_idx]
    bkgs     = data[labels != sig_idx]

    ntotsig = len(sigs)
    ntotbkg = len(bkgs)
    ntotref = len(refs)
    #ntotsrefs = len(srefs)
    z_emp=0
    nsigs   = np.random.poisson(lam=nsig, size=ntoys)
    nbkgs   = np.random.poisson(lam=nbkg, size=ntoys)
    nrefs   = np.random.poisson(lam=nref, size=ntoys)
    nbrfs   = np.random.poisson(lam=nbkg, size=ntoys)
    for pToy in range(ntoys):
        sigidx  = np.random.choice(ntotsig, size=nsigs[pToy], replace=False)
        bkgidx  = np.random.choice(ntotbkg, size=nbkgs[pToy], replace=False)
        refidx  = np.random.choice(ntotref, size=nrefs[pToy], replace=False)
        brfidx  = np.random.choice(ntotref, size=nbkgs[pToy], replace=False) #note to be accurate thsi should be ref, but statisically correct is bkg (its just cheating)
        #srfidx  = np.random.choice(ntotsref,size=nsigs[pToy], replace=False) #note to be accurate thsi should be ref, but statisically correct is bkg (its just cheating)
        sig     = sigs[sigidx]
        bkg     = bkgs[bkgidx]
        ref     = refs[refidx]
        #brf     = bkgs[brfidx] # in the long run we change this to ref
        brf     = refs[brfidx] # in the long run we change this to ref
        ref_label=refs_label[refidx]
        #srf     = srfs[srfidx]
        #srf_label=torch.ones(srf.shape)*sig_idx
        #for pMetric in metrics: #just one for now, otherwise t_sig/t_ref have to be fixed
        if iOption == 0:
            dist,_    = mahalanobis_dist(torch.cat((sig,bkg)),ref,ref_label,plot=False,fit=False)
            ref_dist,_= mahalanobis_dist(brf,ref,ref_label,plot=False,fit=False)
        else:
            #totref=torch.cat((ref,srf))
            #totrefl=torch.cat((ref_label,srf_label))
            #dist     = maxlikelihood(torch.cat((sig,bkg)),model,model_labels,sig_idx,nsig,nbkg)
            dist     = ksscore(torch.cat((sig,bkg)),ref,ref_label,sig_idx)
            ref_dist = ksscore(brf,ref,ref_label,sig_idx)
        t_sig.append(dist)
        t_ref.append(ref_dist)
        ts, tr = np.array(t_sig), np.array(t_ref)
    if iOption == 0:
        if plot:
            bins=np.linspace(np.min(tr)*0.8,np.max(tr)*1.2,20)
            x   = 0.5*(bins[1:]+bins[:-1])
            trvals,_ = np.histogram(tr,bins=bins)
            tsvals,_ = np.histogram(ts,bins=bins)
            plt.errorbar(x,tsvals/np.sum(tsvals),yerr=np.sqrt(tsvals)/np.sum(tsvals),alpha=0.5,marker='.',drawstyle='steps-mid',label="Sig+bkg")
            plt.errorbar(x,trvals/np.sum(trvals),yerr=np.sqrt(trvals)/np.sum(trvals),alpha=0.5,marker='.',drawstyle='steps-mid',label="bkg")
            plt.xlabel("t")
            plt.show()
        z_as=zscore(ts,tr,plot)
        z_emp=zemp(ts,tr,plot)
    else:
        if plot:
            bins=np.linspace(np.max(ts)*(-2.2),np.max(ts)*2.2,20)
            x   = 0.5*(bins[1:]+bins[:-1])
            trvals,_ = np.histogram(tr,bins=bins)
            tsvals,_ = np.histogram(ts,bins=bins)
            plt.errorbar(x,tsvals,yerr=np.sqrt(tsvals),alpha=0.5,marker='.',drawstyle='steps-mid',label="Sig+bkg")
            plt.errorbar(x,trvals,yerr=np.sqrt(trvals),alpha=0.5,marker='.',drawstyle='steps-mid',label="bkg")
            plt.xlabel("t")
            plt.show()
        z_emp = zemp(ts,tr,plot)
        z_as  = np.median(np.array(t_sig))
        #print("z_emp",z_emp,"z_as",z_as)
    return z_as,z_emp
    #z_as, z_emp = plot_2distribution_new(ts, tr, df=np.median(ts), xmin=np.min(tr)-1, xmax=np.max(tr)+1, #ymax=0.03, 
    #                   nbins=8, save=False, output_path='./', Z_print=[1.645,2.33],
    #                   label1='REF', label2='DATA', save_name='', print_Zscore=True)
    #return z_as,z_emp

def z_yield(data,labels,ref,ref_labels,iskip,iNb=1000,iNr=10000,iMin=0,iMax=300,iNbins=11,ntoys=1000,plot=True,iOption=0): #0 maha, 1 maxlikelihood (1D)
    sig_yield = np.linspace(iMin,iMax,iNbins) 
    z_as=[]; z_emp=[]
    for pYield in sig_yield: 
        pZ_as,pZ_emp = run_toy(pYield, iNb, iNr,data,labels,ref,ref_labels,iskip,ntoys=ntoys,plot=False,iOption=iOption)
        z_as.append(pZ_as)
        z_emp.append(pZ_emp)

    if plot:
        plt.plot(sig_yield,z_as)
        plt.plot(sig_yield,z_emp)
        plt.show()
    z_emp = np.array(z_emp)
    z_empmax=np.max(z_emp)
    z_emp_out = np.where( z_emp == z_empmax, z_emp*5, z_emp)
    return sig_yield,np.min(np.vstack((z_as,z_emp_out)),axis=0)
#from GENutils import *
#from ANALYSISutils import *

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)
    
    L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y,iPrint=False):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        if torch.isnan(XX + XY + YY):
            return 0
        if iPrint:
            print("XX:",XX,"XY:",XY,"YY:",YY,"X:",X,"Y:",Y,)
        return (XX - 2 * XY + YY)/(XX + YY + 1e-8)
    
def train_generic_datamc(inepochs,itrainloader,imodel,icriterion,ioptimizer,iCorrectData=False,iMMD=True):
    losses = []
    mmdLoss = MMDLoss()
    for epoch in range(inepochs):#tqdm(range(inepochs)):
        imodel.train()
        epoch_loss = []
        for tmp , labelsd in itrainloader:
            batch_data,labels = tmp
            batch_data = batch_data.float()
            
            # Potential to add any augmentation here
            #features       = imodel(batch_data).unsqueeze(1)
            h              = imodel.encoder(batch_data)
            #preds          = imodel.classifier(h)
            loss = 0
            if iCorrectData:
                mc_mask        = (labelsd == 0)
                #data_mask      = (labelsd == 1)
                shifted        = imodel.shifter(h)
                #print(shifted[0:10],h[0:10])
                mc_mask = mc_mask.reshape((labelsd.shape[0],1))*1.
                h              =  h + shifted*mc_mask
            z              = imodel.projector(h)
            #z              = torch.nn.functional.normalize(z,dim=1)
            if iMMD:
                h = torch.nn.functional.normalize(h,dim=1)
                mc   = (h[labelsd == 0])
                data = (h[labelsd == 1])
                #print(len(h[labelsd == 0]),len(h[labelsd == 1]),"!!")
                lMmdloss = mmdLoss(data,mc)*len(labels)#/(400**2)
                #print(lMmdloss)
                loss = loss + lMmdloss
            # Compute SimCLR loss
            z              = torch.nn.functional.normalize(z,dim=1).unsqueeze(1) # normalize the projection for simclr loss
            loss = loss + icriterion(z,labels=labels)
            logits = imodel.classifier(h)
            loss = loss + 0.5*torch.nn.functional.cross_entropy(logits, labels)
            
            
            # Backward pass and optimization
            ioptimizer.zero_grad()
            loss.backward()
            ioptimizer.step()
        
            epoch_loss.append(loss.item())
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        if epoch % 1 == 0:
            print(f'Epoch [{epoch+1}/{inepochs}], Loss: {mean_loss:.4f}')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)

def prepcut(idata,imodel,iLabel=None,cut_threshold=0.5):
        with torch.no_grad():
            i_out = (imodel(idata.float(),embed=True))
            i_out=torch.nn.functional.softmax(imodel.classifier(i_out)).numpy()
        i_maxval=np.max(i_out,1)
        cut_i   = idata[i_maxval > cut_threshold]
        if iLabel is not None:
            cut_l = iLabel[i_maxval > cut_threshold]
        else:
            cut_l = torch.tensor(np.argmax(i_out,axis=1))[i_maxval > cut_threshold]
        cut_ds    = GenericDataset(cut_i, cut_l)
        return cut_ds
    
def train_generic_datamc_prep(inepochs,iTrain,iTrue,iTrainLabel,iModel,iMMD=True,batch_size=1000,cut_threshold=0,temp=0.5):
        cut_mc = prepcut(iTrain, iModel,cut_threshold=cut_threshold,iLabel=iTrainLabel)
        cut_ds = prepcut(iTrue,  iModel,cut_threshold=cut_threshold)
        merger = ConcatWithLabels([cut_mc,cut_ds],[0,1])
        labels = merger._labels
        num_classes = 2
        sampler  = BalancedBatchSampler(labels, batch_size, num_classes)
        loader   = DataLoader(merger,batch_sampler=sampler,num_workers=11)
        criterion = SupervisedSimCLRLoss(temperature=temp)
        optimizer = torch.optim.AdamW(iModel.parameters(),lr=5e-4)
        train_generic_datamc(inepochs,loader,iModel,criterion,optimizer,iCorrectData=False,iMMD=iMMD)
        

def train_generic_datamc_v2(inepochs,itrainloader,imodel,icriterion,ioptimizer,iNSig=3,iMMD=True):
    losses = []
    mmdLoss = MMDLoss()
    for epoch in range(inepochs):#tqdm(range(inepochs)):
        imodel.train()
        epoch_loss = []
        for tmp in itrainloader:
            batch_data,labels = tmp
            batch_data = batch_data.float()
            
            # Potential to add any augmentation here
            #features       = imodel(batch_data).unsqueeze(1)
            h              = imodel.encoder(batch_data)
            #preds          = imodel.classifier(h)
            loss = 0
            if iCorrectData:
                mc_mask        = (labels < 3)
                #data_mask      = (labelsd == 1)
                shifted        = imodel.shifter(h)
                #print(shifted[0:10],h[0:10])
                mc_mask = mc_mask.reshape((labels.shape[0],1))*1.
                h              =  h + shifted*mc_mask
            z              = imodel.projector(h)
            #z              = torch.nn.functional.normalize(z,dim=1)
            if iMMD:
                h = torch.nn.functional.normalize(h,dim=1)
                for pSig in range(iNSig):
                    mc0   = (h[labels == pSig])
                    data0 = (h[labels == (pSig+iNSig)])
                    #print(pSig,mc0,data0)
                    loss  = loss + (mmdLoss(mc0,data0))*len(labels)/iNSig
                    if torch.isnan(loss):
                        print("MMD:",(mmdLoss(data0,mc0,iPrint=True)),pSig)#,mc0,bmc,data0,bdata)
            # Compute SimCLR loss
            z              = torch.nn.functional.normalize(z,dim=1).unsqueeze(1) # normalize the projection for simclr loss
            loss = loss + icriterion(z,labels=(labels % 3))
            #logits = imodel.classifier(h)
            #loss = loss + 0.5*torch.nn.functional.cross_entropy(logits, (labels % 3))
            
            
            # Backward pass and optimization
            ioptimizer.zero_grad()
            loss.backward()
            ioptimizer.step()
        
            epoch_loss.append(loss.item())
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        if epoch % 1 == 0:
            print(f'Epoch [{epoch+1}/{inepochs}], Loss: {mean_loss:.4f}')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)
