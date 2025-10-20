import torch
import numpy as np
import copy

class JetCLR_Augmenter:
    def __init__(self,num_augs,rotate=True,split=True,distort=True,log_pt=False,mask_particles=False):
        self.num_augs = num_augs
        self.rotate = rotate
        self.split = split
        self.distort = distort
        self.log_pt = log_pt
        self.mask_particles = mask_particles
        print(f"Initiating JetCLR_Augmenter with {num_augs} augmentations, rotate={rotate}, split={split}, distort={distort}, log_pt={log_pt}, mask={mask_particles}")

    def augment(self,data):
        aug_data = []
        for i in range(self.num_augs-1): # the original data point is one of the "augmentations"
            augmented = [{k: v.copy() if type(v) == np.ndarray else copy.deepcopy(v) for k, v in d.items()} for d in data]
            if self.rotate:
                augmented = self.rotate_constits(augmented)
            if self.split:
                augmented = self.split_constits(augmented)
            if self.distort:
                augmented = self.distort_constits(augmented)
            if self.mask_particles:
                augmented = self.mask(augmented)
            aug_data.append(augmented)
        
        #for i in range(len(data)):
        #    for k in data[i].keys():
        #        data[i][k] = [data[i][k]]
        #        for aug in aug_data:
        #            data[i][k].append(aug[i][k])

        #return data
        return [data] + aug_data
    
    def mask(self,data,frac=0.1):
        inputs, labels, observers = data
        pts = inputs['pf_pTs'][0,:]
        mask = inputs['pf_mask'][0,:]
        num_parts = len(mask[mask==1])
        mask_probabilities = 1.0/(pts[mask==1] + 1e-6)
        mask_probabilities /= mask_probabilities.sum()
        to_mask = np.random.choice(np.arange(num_parts),size=int(frac*num_parts),replace=False,p=mask_probabilities)
        inputs['pf_mask'][0,to_mask] = 0
        
        return inputs, labels, observers


    def rotate_constits(self,data):
        """
        Apply random rotation to the constituents in a jet

        Args:
            data: (inputs,labels,observers) -- data returned by the dataloader; should be batched torch tensors
        Returns:
            data: (inputs,labels,observers) -- data with rotated constituents

        """
        inputs, labels, observers = data

        theta = np.random.rand() * 2 * np.pi
        # Create rotation matrices for each angle in the batch
        mtx = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        # apply rotation matrices to the eta-phi coordinates
        inputs['pf_features'][1:,:] = np.einsum('ij,jk->ik',mtx,inputs['pf_features'][1:,:])

        # propagate the rotation to the px, py, pz of the four vectors
        #new_eta = inputs['pf_features'][1,:] + observers['jet_eta']
        #new_phi = inputs['pf_features'][2,:] + observers['jet_phi']
        #inputs['pf_vectors'][0,:] = inputs['pf_pTs'][0,:]*np.cos(new_phi) # px
        #inputs['pf_vectors'][1,:] = inputs['pf_pTs'][0,:]*np.sin(new_phi) # py
        #inputs['pf_vectors'][2,:] = inputs['pf_pTs'][0,:]*np.sinh(new_eta) # pz
        
        return inputs, labels, observers

    def split_constits(self,data,frac=0.1):
        inputs, labels, observers = data
        npart = inputs['pf_features'].shape[1]

        n_zeros = np.count_nonzero(inputs['pf_mask'][0,:]==0)
        nreal = npart - n_zeros
        
        if n_zeros == 0:
            return inputs, labels, observers
        
        nfill = min(int(frac*nreal), n_zeros)
        split_indices = np.random.choice(np.arange(npart-n_zeros), size=nfill, replace=False)
        split_fracs = np.random.uniform(0.01, 0.99, size=nfill)
        idx_add = npart - n_zeros
        for k in range(nfill):
            isplit = split_indices[k]
            f = split_fracs[k]
            
            # splitting z (= pT_i/pT_jet)
            if not self.log_pt:
                inputs['pf_features'][0,isplit] = f * inputs['pf_features'][0,isplit]
                inputs['pf_features'][0,idx_add+k] = (1.0-f) * inputs['pf_features'][0,isplit]
            else:
                inputs['pf_features'][0,isplit] = np.log(f**0.7) + inputs['pf_features'][0,isplit]
                inputs['pf_features'][0,idx_add+k] = np.log((1.0-f)**0.7) + inputs['pf_features'][0,isplit]
                #inputs['pf_features'][0,isplit] = 0.7*(np.log(f*inputs['pf_pTs'][0,isplit])-1.7)
                #inputs['pf_features'][0,idx_add+k] = 0.7*(np.log((1.0-f)*inputs['pf_pTs'][0,isplit])-1.7)
            
            
            # splitting px, py, pz
            inputs['pf_vectors'][0,isplit] = f * inputs['pf_vectors'][0,isplit] # px
            inputs['pf_vectors'][1,isplit] = f * inputs['pf_vectors'][1,isplit] # py
            inputs['pf_vectors'][2,isplit] = f * inputs['pf_vectors'][2,isplit] # pz
            inputs['pf_vectors'][3,isplit] = f * inputs['pf_vectors'][3,isplit] # E
            inputs['pf_vectors'][0,idx_add+k] = (1.0-f) * inputs['pf_vectors'][0,isplit] # px
            inputs['pf_vectors'][1,idx_add+k] = (1.0-f) * inputs['pf_vectors'][1,isplit] # py
            inputs['pf_vectors'][2,idx_add+k] = (1.0-f) * inputs['pf_vectors'][2,isplit] # pz
            inputs['pf_vectors'][3,idx_add+k] = (1.0-f) * inputs['pf_vectors'][3,isplit] # E
            # splitting the pTs
            inputs['pf_pTs'][0,isplit] = f * inputs['pf_pTs'][0,isplit]
            inputs['pf_pTs'][0,idx_add+k] = (1.0-f) * inputs['pf_pTs'][0,isplit]
        
        # set the mask to 1.0 for the new particles
        inputs['pf_mask'][0,:nreal+nfill] = 1

        return inputs, labels, observers

    def distort_constits(self,data,pt_scale=0.1):
        """
        Apply random distortions to the constituents in a jet

        Args:
            data: (inputs,labels,observers) -- data returned by the dataloader; should be batched torch tensors
            pt_scale: 0.1 -- 100 MeV for distortions drawn from N(0,100MeV/pT)
        Returns:
            data: (inputs,labels,observers) -- data with distorted constituents

        """
        inputs, labels, observers = data
        npart = inputs['pf_features'].shape[1]
        
        eta_shifts = np.random.randn(npart) * pt_scale / inputs['pf_pTs'][0,:]
        phi_shifts = np.random.randn(npart) * pt_scale / inputs['pf_pTs'][0,:]
        
        # apply shifts to deta and dphi
        inputs['pf_features'][1,:] += eta_shifts
        inputs['pf_features'][2,:] += phi_shifts

        # propagate the shifts to the px, py, pz of the four vectors
        #new_eta = inputs['pf_features'][1,:] + observers['jet_eta']
        #new_phi = inputs['pf_features'][2,:] + observers['jet_phi']
        #inputs['pf_vectors'][0,:] = inputs['pf_pTs'][0,:]*np.cos(new_phi) # px
        #inputs['pf_vectors'][1,:] = inputs['pf_pTs'][0,:]*np.sin(new_phi) # py
        #inputs['pf_vectors'][2,:] = inputs['pf_pTs'][0,:]*np.sinh(new_eta) # pz

        return inputs, labels, observers

class Mask_Augmenter:
    def __init__(self,num_augs,frac):
        self.num_augs = num_augs
        self.frac = frac
    
    def augment(self,data):
        aug_data = []
        for i in range(self.num_augs-1): # the original data point is one of the "augmentations"
            augmented = [{k: v.copy() if type(v) == np.ndarray else copy.deepcopy(v) for k, v in d.items()} for d in data]
            augmented = self.mask_constits(augmented)
            aug_data.append(augmented)
        return [data] + aug_data


    def mask_constits(self,data):
        """
        Randomly mask `frac` percent of the constituents in a jet

        Args:
            data: (inputs,labels,observers) -- data returned by the dataloader; should be batched torch tensors
        Returns:
            data: (inputs,labels,observers) -- data with rotated constituents

        """
        inputs, labels, observers = data
        
        npart = inputs['pf_features'].shape[1]
        n_zeros = np.count_nonzero(inputs['pf_mask'][0,:]==0)
        nreal = npart - n_zeros

        nmask = int(self.frac * nreal)
        indices = np.arange(npart)[inputs['pf_mask'][0,:]==1]
        mask_indices = np.random.choice(indices, size=nmask, replace=False)
        inputs['pf_mask'][0,mask_indices] = 0
        
        return inputs, labels, observers

