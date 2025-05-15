import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#
import sys
sys.path.append('../../')
import data.datasets as datasets
import data.data_utils as dutils
import argparse


def savetoys(iLabel,iSkip,iNToys,mc_out,mc_lab,iSMax=100,iNbins=21,iOption=2):
    if os.path.exists(iLabel+"maha_toys_disc.npz"):
        data = np.load(iLabel+"maha_toys_disc.npz")
        data_dict = {k: data[k] for k in data.files}
    else:
        data_dict={}
    #MH distance
    lOption=2
    xy1,zscore1,zscore1e=dutils.z_yield  (mc_out,mc_lab,mc_out,mc_lab,iSkip,ntoys=iNToys,iNb=2000,iNr=10000,plot=False,iMin=0,iMax=iSMax,iNbins=iNbins,iOption=lOption)
    if iOption == 0:
        data_dict["toymc"]   = zscore1
        data_dict["toymce"]  = zscore1e
        data_dict["toyxy"]   = xy1
        #data_dict["mc_out"]  = mc_out
        #data_dict["mc_lab"]  = mc_lab
    elif iOption == 1:
        data_dict["toyfmc"]   = zscore1
        data_dict["toyfmce"]  = zscore1e
        data_dict["fmc_out"]  = mc_out
        data_dict["fmc_lab"]  = mc_lab
    else :
        data_dict["toydmc"]   = zscore1
        data_dict["toydmce"]  = zscore1e
        #data_dict["dmc_out"]  = mc_out
        #data_dict["dmc_lab"]  = mc_lab
 
    np.savez(iLabel+"maha_toys_disc.npz", **data_dict)


def trainDisc(iData,iLabels,iSkip,batch_size=1000,ntrain=10000,num_epochs=30,last_dim=16):
        input_dim  = iData.shape[1]
        labels=iLabels
        train_nosig_pre  = iData[iLabels != iSkip]
        train_sig_pre    = iData[iLabels == iSkip]
        labels_nosig_pre = iLabels[iLabels != iSkip]
        labels_sig_pre   = iLabels[iLabels == iSkip]
        ###
        print(len(train_nosig_pre))
        tr_nosigidx_sel = np.random.choice(len(train_nosig_pre), size=100000, replace=False)
        tr_nosigidx_rem = np.setdiff1d    (np.arange(len(train_nosig_pre)), tr_nosigidx_sel)
        train_nosig     = torch.tensor(train_nosig_pre[tr_nosigidx_sel])
        test_nosig      = torch.tensor(train_nosig_pre[tr_nosigidx_rem])
        lab_nosig       = torch.tensor(labels_nosig_pre[tr_nosigidx_rem])
        ###
        tr_sigidx_sel = np.random.choice(len(train_sig_pre), size=30000, replace=False)
        tr_sigidx_rem = np.setdiff1d    (np.arange(len(train_sig_pre)), tr_sigidx_sel)
        train_sig     = torch.tensor(train_sig_pre[tr_sigidx_sel])
        test_sig      = torch.tensor(train_sig_pre[tr_sigidx_rem])
        lab_sig       = torch.tensor(labels_sig_pre[tr_sigidx_rem])
        ###
        print("test:",len(train_sig),len(train_nosig_pre),iData.shape,"!!")
        ntrain=np.max([ntrain,len(train_sig)])

        ridx=torch.randperm(len(train_nosig))[:ntrain]
        train_nosig1=train_nosig[ridx]
        train_comb  = torch.cat((train_nosig1,train_sig[:ntrain]))
        train_lab   = torch.cat((torch.zeros(ntrain),torch.ones(ntrain)))
        dset=dutils.GenericDataset(train_comb,train_lab.long())
        disc_trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
        disc_model=dutils.train_disc(num_epochs,disc_trainloader,input_dim,last_dim=last_dim,output_dim=1)

        test_comb  = torch.cat((test_nosig,test_sig))
        test_lab   = torch.cat((lab_nosig,lab_sig))
        dset=dutils.GenericDataset(test_comb.long(),test_lab.long())

        mc_lab=test_lab
        with torch.no_grad():
            mc_out = (disc_model(test_comb.float()))
        
        return disc_model,mc_out,mc_lab

def main():
    parser = argparse.ArgumentParser(description='A simple example of using argparse')
    parser.add_argument('--seed',  dest='seed'    ,type=int,default=1)
    parser.add_argument('--nsigs', dest='nsigs'   ,type=int,default=4)
    parser.add_argument('--ndisc', dest='ndisc'   ,type=int,default=4)
    parser.add_argument('--nrand', dest='nrand'   ,type=int,default=1)
    parser.add_argument('--embed', dest='embed'   ,type=int,default=4)
    parser.add_argument('--ntrain', dest='ntrain', type=int,default=10000)
    parser.add_argument('--nbins',  dest='nbins',  type=int,default=31)
    parser.add_argument('--ntoys',  dest='ntoys',  type=int,default=100)
    parser.add_argument('--nepochs', dest='nepochs',  type=int,default=50)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + 1000)
    
    embed_dim=args.embed
    nsigs=args.nsigs
    ndisc=args.ndisc
    nrand=args.nrand
    nepochs=args.nepochs
    nj_train   = args.ntrain
    nj_valid   = args.ntrain
    nj_testy   = args.ntrain*5
    sigmax     = (2000*0.1)
    skip       = (nsigs-1)
    labelid=""
    dir="/eos/cms/store/user/pharris/anom2/"
    labelid='sig'+str(nsigs)+'_disc'+str(ndisc)+'_rand'+str(nrand)+'_seed'+str(args.seed)
    file=np.load("model_"+labelid+"maha_toys_space.npz") #model_sig4_disc4_rand10_seed0maha_toys_space.npz
    mc_lab =file["mc_lab"]
    mc_out =file["mc_out"]    
    fmc_out=file["fmc_out"]

    model_base, mc_out1, mc_lab1=trainDisc(mc_out,mc_lab,(args.nsigs-1),num_epochs=args.nepochs)
    savetoys("model_"+labelid,skip,args.ntoys,mc_out1,mc_lab1,iNbins=args.nbins,iSMax=sigmax,iOption=0)

    model_base,fmc_out1,fmc_lab1=trainDisc(fmc_out,mc_lab,(args.nsigs-1),num_epochs=args.nepochs)
    savetoys("model_"+labelid,skip,args.ntoys,fmc_out1,fmc_lab1,iNbins=args.nbins,iSMax=sigmax,iOption=1)

    return
    

if __name__ == "__main__":
    main()
