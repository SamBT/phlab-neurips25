import glob, h5py, math, time, os, json, random, yaml, argparse, datetime, time
import os.path
from scipy.stats import norm, expon, chi2, uniform, chisquare
from pathlib import Path
import torch
import numpy as np

import matplotlib as mpl
#mpl.use('Agg')                                                                                       
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

import sys
sys.path.insert(1, './NysMMDutils/')
from test import NysMMDtest
sys.path.insert(1, '../utils/')
import GENutils as gen
import ANALYSISutils as an

from tqdm import tqdm

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

folder_out = config_json["output_directory"]+'/'
print(folder_out)
if not os.path.isdir(folder_out):
    print('mkdir ', folder_out)
    os.makedirs(folder_out)

M = config_json["M"]

N_Bkg = config_json["N_Bkg"]
N_ref = config_json["N_Ref"]
N_Sig = config_json["N_Sig"]
Ntoys = args.toys

# load data
ref_filepath = config_json["ref_filepath"]
data_filepath = config_json["data_filepath"]
anomaly_label = config_json["anomaly_label"]

reference_npz = np.load(ref_filepath)
if data_filepath == ref_filepath:
    data_npz = reference_npz
else:
    data_npz = np.load(data_filepath)

reference_all = reference_npz['data'][reference_npz['labels'] != anomaly_label]
data_all = data_npz['data'][data_npz['labels'] != anomaly_label] if data_filepath != ref_filepath else reference_all
anomaly_all = data_npz['data'][data_npz['labels'] == anomaly_label]
Pois_ON=config_json["Pois_ON"]

mean_all, std_all = np.mean(reference_all, axis=0), np.std(reference_all, axis=0)
std_all[std_all==0] = 1 # avoid zero denominators if any feature is empty
reference_all = gen.standardize(reference_all, mean_all, std_all)
data_all = gen.standardize(data_all, mean_all, std_all)
anomaly_all = gen.standardize(anomaly_all, mean_all, std_all)

# candidate sigma
# you can try others too! 50% corresponds to the third value in NPLM
# and it is standard for MMD
sigma = gen.candidate_sigma(reference_all[:2000, :],
                                        perc=(50))
                        
t_list = []
for toy in tqdm(range(Ntoys)):
    #np.random.seed(toy)
    N_bkg_p, N_sig_p = N_Bkg, N_Sig
    if Pois_ON:
        N_bkg_p = np.random.poisson(lam=N_Bkg, size=1)[0]
        N_sig_p = np.random.poisson(lam=N_Sig, size=1)[0]
        
    idx_ref = np.arange(reference_all.shape[0])
    np.random.shuffle(idx_ref)
    idx_bkg = np.arange(data_all.shape[0])
    np.random.shuffle(idx_bkg)
    idx_sig = np.arange(anomaly_all.shape[0])
    np.random.shuffle(idx_sig)

    if data_filepath == ref_filepath:
        idx_bkg = idx_ref
        bkg_offset = N_ref
    else:
        bkg_offset = 0

    X_test = np.concatenate((anomaly_all[idx_sig[:N_sig_p]], data_all[idx_bkg[bkg_offset:bkg_offset+N_bkg_p]], reference_all[idx_ref[:N_ref]]), axis=0)
        
    test_toy = NysMMDtest(Z=X_test, n=N_bkg_p+N_sig_p, m=N_ref, seed=None, method='uniform', bandwidth=sigma, k=M)
    
    #if not toy%10: print(toy, test_toy)
    
    t_list.append(test_toy)
t_list = np.array(t_list)
print('Total number of toys accumulated ', len(t_list))
tmp_id = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
np.save(f"{config_json['output_directory']}/{tmp_id}_test.npy", t_list)
