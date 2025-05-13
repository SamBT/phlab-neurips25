import glob, h5py, math, time, os, json, random, yaml, argparse, datetime
from scipy.stats import norm, expon, chi2, uniform, chisquare
from pathlib import Path
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

import sys
sys.path.insert(1, '../utils/')
from FLKutils import *
from GENutils import *

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=True)
parser.add_argument('-t', '--toys', type=int, help="number of toys", required=True)

args = parser.parse_args()

# train on GPU?                                                                                                                              
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

json_path = args.jsonfile
with open(json_path, 'r') as jsonfile:
    config_json = json.load(jsonfile)

# configs                                                                                                              \                                                                                                               
M          = config_json["M"]                                                                                            
lam        = config_json["lambda"]                                                                                    
iterations = config_json["iterations"]                                                                                 
flk_sigmas = config_json["falkon_sigmas"]
Ntoys      = args.toys

N_ref      = config_json["N_Ref"]
N_Bkg      = config_json["N_Bkg"]
N_Sig      = config_json["N_Sig"]
z_ratio    = N_Bkg*1./N_ref
is_Pois    = config_json["Pois_ON"]


ref_filepath = config_json["ref_filepath"]
data_filepath = config_json["data_filepath"]
anomaly_label = config_json["anomaly_label"]

folder_out = config_json["output_directory"]+'/'
print(folder_out)
if not os.path.isdir(folder_out):
    print('mkdir ', folder_out)
    os.makedirs(folder_out)


reference_npz = np.load(ref_filepath)
if data_filepath == ref_filepath:
    data_npz = reference_npz
else:
    data_npz = np.load(data_filepath)

reference_all = reference_npz['data'][reference_npz['labels'] != anomaly_label]
data_all = data_npz['data'][data_npz['labels'] != anomaly_label] if data_filepath != ref_filepath else reference_all
anomaly_all = data_npz['data'][data_npz['labels'] == anomaly_label]

# standardize
mean_all, std_all = np.mean(reference_all, axis=0), np.std(reference_all, axis=0)
std_all[std_all==0] = 1 # avoid zero denominators if any feature is empty
reference_all = standardize(reference_all, mean_all, std_all)
data_all = standardize(data_all, mean_all, std_all)
anomaly_all = standardize(anomaly_all, mean_all, std_all)

# compute candidate sigma from reference typical distances   
print('flk_sigma', candidate_sigma(reference_all[:2000, :], 
                                   perc=([1, 5, 10, 25, 50, 75, 90, 95, 99]))
     )

# initialize
tstat_dict = {}
seeds_dict = {}
preds_dict = {}
ref_idx_dict = {}
data_idx_dict = {}
anomaly_idx_dict = {}
seeds_flk_dict = {}
seed_toys = np.arange(Ntoys)*int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)

# training 
for flk_sigma in flk_sigmas:
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=z_ratio,iter=[iterations],seed=None,cpu=cuda)
    t_list = []
    preds_list = []
    ref_idx_list = []
    data_idx_list = []
    anomaly_idx_list = []
    seeds_flk = []
    for i in range(Ntoys):
        # set the seed
        seed = seed_toys[i]
        rng = np.random.default_rng(seed=seed)
        # make poisson statistics if is_Pois==True     
        N_Bkg_Pois = N_Bkg
        N_Sig_Pois = N_Sig
        if is_Pois:
            N_Bkg_Pois = rng.poisson(lam=N_Bkg, size=1)[0]
            N_Sig_Pois = rng.poisson(lam=N_Sig, size=1)[0]
            
        rng = np.random.default_rng(seed=seed)
        # randomly extract data
        idx_bkg = rng.permutation(data_all.shape[0])
        idx_sig = rng.permutation(anomaly_all.shape[0])
        idx_ref = rng.permutation(reference_all.shape[0])
        if data_filepath == ref_filepath:
            idx_bkg = idx_ref
            bkg_offset = N_ref
        else:
            bkg_offset = 0

        data_x = np.concatenate((anomaly_all[idx_sig[:N_Sig_Pois]], data_all[idx_bkg[bkg_offset:bkg_offset+N_Bkg_Pois]]), axis=0)
        data_y = np.ones((data_x.shape[0], 1))
        ref_x = reference_all[idx_ref[:N_ref]]
        ref_y = np.zeros((ref_x.shape[0], 1))
        feature = np.concatenate((data_x, ref_x), axis=0).astype(float)
        target  = np.concatenate((data_y, ref_y), axis=0).astype(float)

        #ref_idx_list.append(idx_ref[:N_ref].reshape(1,-1))
        #data_idx_list.append(idx_bkg[bkg_offset:bkg_offset+N_Bkg_Pois].reshape(1,-1))
        #anomaly_idx_list.append(idx_sig[:N_Sig_Pois].reshape(1,-1))

        # run
        seed_flk = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
        seeds_flk.append(seed_flk)
        t_tmp, preds_tmp = run_toy("", feature, target,  weight=z_ratio, seed=seed_flk,
                              flk_config=flk_config, output_path='./', plot=False, verbose=False, df=10)
        t_list.append(t_tmp)
        preds_list.append(preds_tmp.reshape(1,-1))
        print(t_list[-1])
    tstat_dict[str(flk_sigma)]=np.array(t_list)
    preds_dict[str(flk_sigma)]=np.concatenate(preds_list, axis=0)
    seeds_flk_dict[str(flk_sigma)]=np.array(seeds_flk)
    #ref_idx_dict[str(flk_sigma)]=np.concatenate(ref_idx_list, axis=0)
    #data_idx_dict[str(flk_sigma)]=np.concatenate(data_idx_list, axis=0)
    #anomaly_idx_dict[str(flk_sigma)]=np.concatenate(anomaly_idx_list, axis=0)

# save on temporary file  
tmp_id = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
f = h5py.File('%s/%i_tests.h5'%(folder_out,  tmp_id), 'w')
for flk_sigma in flk_sigmas:
  f.create_dataset(str(flk_sigma), data=tstat_dict[str(flk_sigma)], compression='gzip')
  f.create_dataset('seed_flk_%s'%(str(flk_sigma)), data=seeds_flk_dict[str(flk_sigma)], compression='gzip')
  f.create_dataset('preds_%s'%(str(flk_sigma)), data=preds_dict[str(flk_sigma)], compression='gzip')
  #f.create_dataset('ref_evt_idx_%s'%(str(flk_sigma)), data=ref_idx_dict[str(flk_sigma)], compression='gzip')
  #f.create_dataset('data_evt_idx_%s'%(str(flk_sigma)), data=data_idx_dict[str(flk_sigma)], compression='gzip')
  #f.create_dataset('anomaly_evt_idx_%s'%(str(flk_sigma)), data=anomaly_idx_dict[str(flk_sigma)], compression='gzip')
f.create_dataset('seed_toy', data=seed_toys, compression='gzip')
f.close()
