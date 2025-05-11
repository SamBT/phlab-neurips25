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

def edgeid(iDS):
    edgid = 0
    for pVar in range(iDS.ndisc):
        pChoice=iDS.choice[pVar][iDS.skip]
        tmp=[]
        for vals in iDS.choice[pVar]:
            tmp.append(iDS.mins[vals])
        tmp=np.array(tmp)
        if iDS.mins[pChoice] == np.min(tmp) or iDS.mins[pChoice] == np.max(tmp):
            edgid = edgid + 1
    return edgid   

def savetoys(iLabel,iSkip,iNToys,mc_out,da_out,mc_lab,da_lab,mc_raw,da_raw,edgeid,iSMax=100,iNbins=21,iRaw=True):
    #MH distance
    xy1,zscore1=dutils.z_yield  (mc_out,mc_lab,       mc_out,  mc_lab,  iSkip,ntoys=iNToys,iNb=10000,iNr=20000,plot=False,iMin=0,iMax=iSMax,iNbins=iNbins)
    xy1d,zscore1d=dutils.z_yield(da_out,da_lab,       mc_out,  mc_lab,  iSkip,ntoys=iNToys,iNb=10000,iNr=20000,plot=False,iMin=0,iMax=iSMax,iNbins=iNbins)
    if not iRaw:
        iNToys = 1
    xy2,zscore2=dutils.z_yield  (mc_raw,mc_lab,       mc_raw,  mc_lab,  iSkip,ntoys=iNToys,iNb=10000,iNr=20000,plot=False,iMin=0,iMax=iSMax,iNbins=iNbins)
    xy2d,zscore2d=dutils.z_yield(da_raw,da_lab,       mc_raw,  mc_lab,  iSkip,ntoys=iNToys,iNb=10000,iNr=20000,plot=False,iMin=0,iMax=iSMax,iNbins=iNbins)
    
    np.savez(iLabel+"maha_toys_space.npz", toyxy=xy1, toymc=zscore1, toydata=zscore1d, toyrawmc=zscore2,toyrawdata=zscore2d,mc_out=mc_out,da_out=da_out,mc_lab=mc_lab,da_lab=da_lab,edgeid=edgeid)


def main():
    parser = argparse.ArgumentParser(description='A simple example of using argparse')
    parser.add_argument('--seed',  dest='seed'    ,type=int,default=1)
    parser.add_argument('--nsigs', dest='nsigs'   ,type=int,default=4)
    parser.add_argument('--ndisc', dest='ndisc'   ,type=int,default=4)
    parser.add_argument('--nrand', dest='nrand'   ,type=int,default=1)
    parser.add_argument('--embed', dest='embed'   ,type=int,default=4)
    parser.add_argument('--ntrain', dest='ntrain', type=int,default=10000)
    parser.add_argument('--nbins',  dest='nbins',  type=int,default=21)
    parser.add_argument('--ntoys',  dest='ntoys',  type=int,default=1000)
    parser.add_argument('--nepochs', dest='nepochs',  type=int,default=15)

    args = parser.parse_args()
    np.random.seed(args.seed)
    embed_dim=args.embed
    nsigs=args.nsigs
    ndisc=args.ndisc
    nrand=args.nrand
    nepochs=args.nepochs
    nj_train   = args.ntrain
    nj_valid   = args.ntrain
    nj_testy   = args.ntrain
    sigmax     = (nj_train*0.02)
    id='disc'+str(ndisc)+'_rand'+str(nrand)+'_seed'+str(args.seed)

    tjds       = datasets.FlatDataset(nsigs,ndisc,nj_train,nj_valid,nj_testy,nrand)
    torch.save(tjds,'flat_data_'+id+'.pt')
    edgid      = edgeid(tjds)
    
    model_base,otrain,da_out,mc_out,mc_lab,da_lab=tjds.trainQuick(embed_dim=4,num_epochs=nepochs,temp=0.01,plot=False)
    torch.save(tjds,'model_'+id+'.pt')                   

    #MH distance
    skip=tjds.skip
    intoys=args.ntoys
    savetoys("model_"+id,tjds.skip,args.ntoys,mc_out,da_out,mc_lab,da_lab,tjds.test_data,tjds.trut_data,edgid,iNbins=args.nbins,iSMax=sigmax)
    
    #now correcting with weak labels
    model_base,otrain,da_out,mc_out,mc_lab,da_lab=tjds.trainQuickDataMC(imodel=model_base,embed_dim=embed_dim,num_epochs=nepochs,temp=0.01,plot=False)
    torch.save(tjds,'modelcor_'+id+'.pt')
    savetoys("modelcor_"+id,tjds.skip,args.ntoys,mc_out,da_out,mc_lab,da_lab,tjds.test_data,tjds.trut_data,edgid,iNbins=args.nbins,iRaw=False,iSMax=sigmax)
    
    #now correcting domain shift
    model_base,otrain,da_out,mc_out,mc_lab,da_lab=tjds.trainQuickDataMC(imodel=model_base,embed_dim=embed_dim,num_epochs=nepochs,temp=0.01,iMMD=True,plot=False)
    torch.save(tjds,'modelcor_mmd_'+id+'.pt')
    savetoys("modelcor_mmd_"+id,tjds.skip,args.ntoys,mc_out,da_out,mc_lab,da_lab,tjds.test_data,tjds.trut_data,edgid,iNbins=args.nbins,iRaw=False,iSMax=sigmax)
    
if __name__ == "__main__":
    main()


#Some random cdoe for latter
#train_sfree   = tjds.train_data  [tjds.train_labels != (nsigs-1)]
#true_sfree    = torch.cat((tjds.true_data   [tjds.true_labels  != (nsigs-1)],tjds.true_data   [tjds.true_labels  == (nsigs-1)][0:0.01*nj_train))#adding 1percent contamination
#train_sfrel   = tjds.train_labels[tjds.train_labels != (nsigs-1)]
#dutils.train_generic_datamc_prep(5,train_sfree,true_sfree,train_sfrel,model_base,cut_threshold=0.2,temp=0.5,iMMD=False)
