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

def savetoys(iLabel,iSkip,iNToys,mc_out,da_out,mc_lab,da_lab,da_weights,mc_weights,iOption=0):
    if os.path.exists(iLabel+"nplm_toys_space_spark_4d.npz"):
        data = np.load(iLabel+"nplm_toys_space_spark_4d.npz")
        data_dict = {k: data[k] for k in data.files}
    else:
        data_dict={}

    nref=40000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #da_out = da_out.to(device)
    #mc_out = mc_out.to(device)
    #da_lab = da_lab.to(device)
    #mc_lab = mc_lab.to(device)
    #da_weights = da_weights.to(device)
    #mc_weights = mc_weights.to(device)
    ts,tb=dutils.run_realistic_toy(nref,da_out,da_lab,mc_out,mc_lab,iSkip,data_weights=da_weights,model_weights=mc_weights,ntoys=iNToys,plot=False,width=30.0,splitmass=True) #3.0
    data_dict["toy_sig"]  = ts
    data_dict["toy_bkg"]  = tb
    np.savez(iLabel+"nplm_toys_ctr_space_tight.npz", **data_dict)


def main():
    parser = argparse.ArgumentParser(description='A simple example of using argparse')
    parser.add_argument('--seed',  dest='seed'    ,type=int,default=1)
    parser.add_argument('--ntoys',  dest='ntoys',  type=int,default=2)
    parser.add_argument('--model',  dest='model',  type=str,default='higgs_raw_tight_spark.npz')
    parser.add_argument('--niters', dest='niters',  type=int,default=1)

    args = parser.parse_args()
    np.random.seed(args.seed*args.niters)
    torch.manual_seed(args.seed*args.niters + 1000)
    
    dir="/eos/cms/store/user/pharris/anom2/"
    dir=""
    #labelid="higgs_base_disc_ctr.npz"
    labelid=args.model
    file_path=dir+labelid
    file=np.load(file_path) #model_sig4_disc4_rand10_seed0maha_toys_space.npz                                                                                                                        

    tdata         = torch.tensor(file["toy_data_out"])
    tdata_labels  = torch.tensor(file["toy_data_label"])
    tdata_weights = torch.tensor(file["toy_data_weights"])

    mdata         = torch.tensor(file["toy_mc_out"])
    mdata_labels  = torch.tensor(file["toy_mc_label"])
    mdata_weights = torch.tensor(file["toy_mc_weights"])
    mdata = mdata.squeeze(2)
    tdata = tdata.squeeze(2)
    print(mdata.shape,tdata.shape,"!!!")
    
    id='higgs_seed'+str(args.seed*args.niters)
    np.random.seed(args.seed*args.niters)
    
    #MH distance
    skip=1
    intoys=args.ntoys
    savetoys("model_"+id,skip,args.ntoys,mdata,tdata,mdata_labels,tdata_labels,tdata_weights,mdata_weights,iOption=2)
    return
    
if __name__ == "__main__":
    main()

