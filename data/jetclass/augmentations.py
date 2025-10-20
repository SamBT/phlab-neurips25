import torch
import numpy as np

def JetCLR_Augmentations(data,num_augs,rotate=True,split=True,distort=True):
    """Apply JetCLR augmentations to the input data.

    Args:
        data (dict): A dictionary containing jet data with keys 'pf_features' and 'pf_pTs'.

    Returns:
        dict: A dictionary containing two augmented views of the input data.
    """
    aug_data = []
    inputs,labels,observers = data
    for i in range(num_augs):
        augmented = {k: v.clone() for k, v in inputs.items()}, {k: v.clone() for k, v in labels.items()}, {k: v.clone() for k, v in observers.items()}
        if rotate:
            augmented = rotate_constits(augmented)
        if split:
            augmented = split_constits(augmented)
        if distort:
            augmented = distort_constits(augmented)
        aug_data.append(augmented)

    return aug_data


def rotate_constits(data):
    """
    Apply random rotation to the constituents in a jet

    Args:
        data: (inputs,labels,observers) -- data returned by the dataloader; should be batched torch tensors
    Returns:
        data: (inputs,labels,observers) -- data with rotated constituents

    """
    inputs, labels, observers = data
    aug_inputs = {k: v.clone() for k, v in inputs.items()}
    bs = aug_inputs['pf_features'].shape[0]

    thetas = torch.rand(bs) * 2 * torch.pi
    # Create rotation matrices for each angle in the batch
    cos_theta = torch.cos(thetas)
    sin_theta = torch.sin(thetas)
    zeros = torch.zeros_like(thetas)

    # Build 2x2 rotation matrices for each sample in the batch
    # Shape will be [bs, 2, 2]
    rot_matrices = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta, cos_theta], dim=1)
    ], dim=1).to(aug_inputs['pf_features'])

    # apply rotation matrices to the eta-phi coordinates
    aug_inputs['pf_features'][:,1:,:] = torch.einsum('bij,bjk->bik',
                                                     rot_matrices,
                                                     aug_inputs['pf_features'][:,1:,:])

    # propagate the rotation to the px, py, pz of the four vectors
    new_eta = aug_inputs['pf_features'][:,1,:] + observers['jet_eta'].view(bs,1)
    new_phi = aug_inputs['pf_features'][:,2,:] + observers['jet_phi'].view(bs,1)
    aug_inputs['pf_vectors'][:,0,:] = aug_inputs['pf_pTs'][:,0,:]*torch.cos(new_phi) # px
    aug_inputs['pf_vectors'][:,1,:] = aug_inputs['pf_pTs'][:,0,:]*torch.sin(new_phi) # py
    aug_inputs['pf_vectors'][:,2,:] = aug_inputs['pf_pTs'][:,0,:]*torch.sinh(new_eta) # pz
    
    return aug_inputs, labels, observers

def split_constits(data):
    inputs, labels, observers = data
    aug_inputs = {k: v.clone() for k, v in inputs.items()}
    bs = aug_inputs['pf_features'].shape[0]
    npart = aug_inputs['pf_features'].shape[2]

    for ib in range(bs):
        n_zeros = torch.count_nonzero(aug_inputs['pf_mask'][ib,0,:]==0).item()
        if n_zeros == 0:
            continue
        nfill = min(npart-n_zeros, n_zeros)
        split_indices = np.random.choice(np.arange(npart-n_zeros), size=nfill, replace=False)
        split_fracs = np.random.uniform(0.0, 1.0, size=nfill)
        idx_add = npart - n_zeros
        for k in range(nfill):
            isplit = split_indices[k]
            f = split_fracs[k]
            # splitting z (= pT_i/pT_jet)
            aug_inputs['pf_features'][ib,0,isplit] = f * aug_inputs['pf_features'][ib,0,isplit]
            aug_inputs['pf_features'][ib,0,idx_add+k] = (1.0-f) * aug_inputs['pf_features'][ib,0,isplit]
            # splitting px, py, pz
            aug_inputs['pf_vectors'][ib,0,isplit] = f * aug_inputs['pf_vectors'][ib,0,isplit] # px
            aug_inputs['pf_vectors'][ib,1,isplit] = f * aug_inputs['pf_vectors'][ib,1,isplit] # py
            aug_inputs['pf_vectors'][ib,2,isplit] = f * aug_inputs['pf_vectors'][ib,2,isplit] # pz
            aug_inputs['pf_vectors'][ib,3,isplit] = f * aug_inputs['pf_vectors'][ib,3,isplit] # E
            aug_inputs['pf_vectors'][ib,0,idx_add+k] = (1.0-f) * aug_inputs['pf_vectors'][ib,0,isplit] # px
            aug_inputs['pf_vectors'][ib,1,idx_add+k] = (1.0-f) * aug_inputs['pf_vectors'][ib,1,isplit] # py
            aug_inputs['pf_vectors'][ib,2,idx_add+k] = (1.0-f) * aug_inputs['pf_vectors'][ib,2,isplit] # pz
            aug_inputs['pf_vectors'][ib,3,idx_add+k] = (1.0-f) * aug_inputs['pf_vectors'][ib,3,isplit] # E
            # splitting the pTs
            aug_inputs['pf_pTs'][ib,0,isplit] = f * aug_inputs['pf_pTs'][ib,0,isplit]
            aug_inputs['pf_pTs'][ib,0,idx_add+k] = (1.0-f) * aug_inputs['pf_pTs'][ib,0,isplit]
    
    # set the mask to 1.0 for everything, since we've now totally filled the vectors
    aug_inputs['pf_mask'] = torch.ones_like(aug_inputs['pf_mask'])

    return aug_inputs, labels, observers

def distort_constits(data,pt_scale=0.1):
    """
    Apply random distortions to the constituents in a jet

    Args:
        data: (inputs,labels,observers) -- data returned by the dataloader; should be batched torch tensors
        pt_scale: 0.1 -- 100 MeV for distortions drawn from N(0,100MeV/pT)
    Returns:
        data: (inputs,labels,observers) -- data with distorted constituents

    """
    inputs, labels, observers = data
    aug_inputs = {k: v.clone() for k, v in inputs.items()}
    bs = aug_inputs['pf_features'].shape[0]
    npart = aug_inputs['pf_features'].shape[2]
    
    eta_shifts = torch.randn(bs,npart) * pt_scale / aug_inputs['pf_pTs'][:,0,:]
    phi_shifts = torch.randn(bs,npart) * pt_scale / aug_inputs['pf_pTs'][:,0,:]
    
    # apply shifts to deta and dphi
    aug_inputs['pf_features'][:,1,:] += eta_shifts
    aug_inputs['pf_features'][:,2,:] += phi_shifts

    # propagate the shifts to the px, py, pz of the four vectors
    new_eta = aug_inputs['pf_features'][:,1,:] + observers['jet_eta'].view(bs,1)
    new_phi = aug_inputs['pf_features'][:,2,:] + observers['jet_phi'].view(bs,1)
    aug_inputs['pf_vectors'][:,0,:] = aug_inputs['pf_pTs'][:,0,:]*torch.cos(new_phi) # px
    aug_inputs['pf_vectors'][:,1,:] = aug_inputs['pf_pTs'][:,0,:]*torch.sin(new_phi) # py
    aug_inputs['pf_vectors'][:,2,:] = aug_inputs['pf_pTs'][:,0,:]*torch.sinh(new_eta) # pz

    return aug_inputs, labels, observers


