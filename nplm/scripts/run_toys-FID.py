import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.stats import norm
import sys
import os
sys.path.insert(1, '../utils/')
from ANALYSISutils import *
import GENutils as gen
import json
import argparse
from tqdm import tqdm
import datetime

def compute_fid(X_1, X_2, eps=1e-6):
    """
    Compute the Fréchet Inception Distance (FID) between two sets of samples.
    
    Parameters:
        X_1: np.ndarray, shape (n_samples_1, n_features)
        X_2: np.ndarray, shape (n_samples_2, n_features)
        eps: small value for numerical stability in covariance
    
    Returns:
        fid: float, the FID score
    """
    # Compute means
    mu_1 = np.mean(X_1, axis=0)
    mu_2 = np.mean(X_2, axis=0)
    
    # Compute covariances
    sigma_1 = np.cov(X_1, rowvar=False) + eps * np.eye(X_1.shape[1])
    sigma_2 = np.cov(X_2, rowvar=False) + eps * np.eye(X_2.shape[1])

    # Compute sqrt of product of covariances
    covmean = sqrtm(sigma_1 @ sigma_2)

    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID
    mean_diff = np.sum((mu_1 - mu_2) ** 2)
    trace_term = np.trace(sigma_1 + sigma_2 - 2 * covmean)

    fid = mean_diff + trace_term
    return fid

OUTPUT_DIRECTORY =  '../experiments_output/'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

parser   = argparse.ArgumentParser()
parser.add_argument('-n','--name',    type=str, help="name of the experiment", required=True)
parser.add_argument('-a','--anomaly-label', type=int, help="anomaly label", required=True)
parser.add_argument('-r','--ref-filepath', type=str, help="reference file path", required=True)
parser.add_argument('-d','--data-filepath', type=str, help="data file path", required=True)
parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", default='toys-NysMMD.py')
parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default=1)
parser.add_argument('-j', '--jobs',    type=int, help="number of jobs submissions",   required=False, default = 100)
parser.add_argument('-s', '--nsig', type=int, help="number of signal events to inject", required=True)
parser.add_argument('-o','--reference-only', action='store_true')
parser.add_argument('--nref',type=int, help="number of reference events", required=False, default=10000)
parser.add_argument('--nbkg',type=int, help="number of background events", required=False, default=2000)
parser.add_argument('-q','--queue', type=str, help="queue to submit jobs to", required=False, default='iaifi_gpu_priority')
args = parser.parse_args()

ntoys = args.toys
njobs = args.jobs

config_json = {
    'ref_filepath': args.ref_filepath,
    'data_filepath': args.data_filepath,
    "N_Ref"   : args.nref,
    "N_Bkg"   : args.nbkg,
    "N_Sig"   : args.nsig,
    "anomaly_label": args.anomaly_label,
    "M": int(np.sqrt(args.nref + args.nbkg)),
    "plot": False,
    "Pois_ON": False,
}

is_pois = ''
if not config_json["Pois_ON"]:
    is_pois = 'NoPois'
ID = f'{args.name}/FID_{is_pois}/'
# problem specs
ID += '/Nref'+str(config_json["N_Ref"])+'_Nbkg'+str(config_json["N_Bkg"])+"_Nsig"+str(config_json["N_Sig"])

# create output folder
config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+ID
if not os.path.exists(config_json["output_directory"]):                                                                   
    os.makedirs(config_json["output_directory"],exist_ok=True)
config_json['pyscript'] = args.pyscript
pyscript = args.pyscript
pyscript_str = args.pyscript.replace('.py', '')
# save json file with problem settings
json_path = create_config_file(config_json, config_json["output_directory"])

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

t_list = []
for toy in tqdm(range(Ntoys)):
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
    Y_test = np.concatenate((np.ones((N_sig_p+N_bkg_p, )), np.zeros((N_ref,))), axis=0)
    test_toy = compute_fid(X_test[Y_test==1], X_test[Y_test==0] )
    #if not toy: print(test_toy)
    t_list.append(test_toy)
t_list = np.array(t_list)
print('Total number of toys accumulated ', len(t_list))
tmp_id = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
np.save(f"{config_json['output_directory']}/{tmp_id}_test.npy", t_list)