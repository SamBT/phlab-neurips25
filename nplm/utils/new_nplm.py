import glob, h5py, math, time, os, json
from scipy.stats import norm, expon, chi2, uniform, chisquare, gamma, beta
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import glob, h5py, math, time, os, json
from scipy.stats import norm, expon, chi2, uniform, chisquare, gamma, beta
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import json
#plt.rcParams["font.family"] = "serif"
#plt.style.use('classic')

def z_to_p(z):
    return norm.sf(z)

def p_to_z(pvals):
    return norm.ppf(1 - pvals)

def Z_score_chi2(t,df):
    sf = chi2.sf(t, df)
    Z  = -norm.ppf(sf)
    return Z

def Z_score_gamma(t, a, loc, scale):
    sf = gamma.sf(t, a, loc, scale)
    Z  = -norm.ppf(sf)
    return Z

def Z_score_norm(t,mu, std):
    sf = norm.sf(t, mu, std)
    Z  = -norm.ppf(sf)
    return Z

def emp_pvalue(ref,t):
    # this definition is such that p=1/(N+1)!=0 if data is the most extreme value
    p = (np.count_nonzero(ref > t)+1) / (len(ref)+1)
    return p

def emp_pvalues(ref,data):
    return np.array([emp_pvalue(ref,t) for t in data])

def return_pvalues(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p_ref[:,idx] = np.transpose([emp_pvalue(np.delete(col,idx2),el) for idx2, el in enumerate(col)])

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data

def min_p(ref,data):
    p_ref, p_data = return_pvalues(ref,data)
    return -np.log(np.min(p_ref,axis=1)), -np.log(np.min(p_data,axis=1))

def avg_p(ref,data):
    p_ref, p_data = return_pvalues(ref,data)
    return -np.log(np.mean(p_ref,axis=1)), -np.log(np.mean(p_data,axis=1))    

def prod_p(ref,data):
    p_ref, p_data = return_pvalues(ref,data)
    return -np.sum(np.log(p_ref),axis=1), -np.sum(np.log(p_data),axis=1)

def power_emp(t_ref,t_data,zalpha=[1,2,3]):
    alpha = z_to_p(zalpha)
    #print(alpha)
    quantiles = np.quantile(t_ref,1-alpha,method='higher')
    total = len(t_data) 
    passed_list = [np.sum(t_data>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [passed/total for passed in passed_list]
    return p_to_z(alpha), power_list, err_list

def z_emp(t_ref,t_data):
    # empirical z score
    t_empirical = np.count_nonzero(t_ref > np.mean(t_data))/len(t_ref)
    if t_empirical==0:
        t_empirical = 1./len(t_ref)
    z_empirical = norm.ppf(1-t_empirical)
    return z_empirical

def ClopperPearson_interval(total, passed, level):
    low_b, up_b = 0.5*(1-level), 0.5*(1+level)
    low_q=beta.ppf(low_b, passed, total-passed+1, loc=0, scale=1)
    up_q=beta.ppf(up_b, passed, total-passed+1, loc=0, scale=1)
    return np.around(passed*1./total-low_q, 5), np.around(up_q-passed*1./total,5)

def get_zScores(t1, t2, df):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!).
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    # t1
    Z1_obs     = Z_score_chi2(np.median(t1), df)
    t1_obs_err = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    Z1_obs_p   = Z_score_chi2(np.median(t1)+t1_obs_err, df)
    Z1_obs_m   = Z_score_chi2(np.median(t1)-t1_obs_err, df)
    
    # t2
    Z2_obs     = Z_score_chi2(np.median(t2), df)
    t2_obs_err = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z2_obs_p   = Z_score_chi2(np.median(t2)+t2_obs_err, df)
    Z2_obs_m   = Z_score_chi2(np.median(t2)-t2_obs_err, df)
    t2_empirical = np.sum(1.*(t1>np.mean(t2)))*1./t1.shape[0]
    empirical_lim = '='
    if t2_empirical==0:
        empirical_lim='>'
        t2_empirical = 1./t1.shape[0]
    t2_empirical_err = t2_empirical*np.sqrt(1./np.sum(1.*(t1>np.mean(t2))+1./t1.shape[0]))
    Z2_empirical = norm.ppf(1-t2_empirical)
    Z2_empirical_m = norm.ppf(1-(t2_empirical+t2_empirical_err))
    Z2_empirical_p = norm.ppf(1-(t2_empirical-t2_empirical_err))
                                          
    return [Z2_obs, Z2_obs_p, Z2_obs_m], [Z2_empirical, Z2_empirical_p, Z2_empirical_m]

def NPLM_combined_pval(t_null, t_alt, rule='min'):
    '''
    t_null: set of toys computed under the null hypothesis 
            with different widths hyper-parameter (shape: [Ntoys, Nwidths] )
    t_alt: set of toys computed under the alternative hypothesis 
           with different widths hyper-parameter (shape: [Ntoys, Nwidths] )
    RETURN:
    - p_ref: numpy array of combined pvalue for the null test (shape: [Ntoys,])
    - p_data: numpy array of combined pvalue for the alternative test (shape: [Ntoys,])
    '''
    if rule=='min':
        p_ref, p_data =  min_p(t_null,t_alt)
    elif rule=='prod':
        p_ref, p_data = prod_p(t_null,t_alt)
    elif rule=='avg':
        p_ref, p_data = avg_p(t_null,t_alt)
    else:
        print("unrecognized rule. Available rules are : min, prod, avg.")
        return
    return p_ref, p_data

def retrieve_data(config_json,seed):
    with open(config_json,'r') as json_file:
        config = json.load(json_file)
    # load the configuration file
    ref_filepath = config['ref_filepath']
    data_filepath = config['data_filepath']
    anomaly_label = config["anomaly_label"]
    N_ref = config['N_Ref']
    N_Bkg = config['N_Bkg']
    N_Sig = config['N_Sig']
    is_Pois = config["Pois_ON"]


    rng1 = np.random.default_rng(seed=seed)
    N_Bkg_Pois = N_Bkg
    N_Sig_Pois = N_Sig
    if is_Pois:
            N_Bkg_Pois = rng1.poisson(lam=N_Bkg, size=1)[0]
            N_Sig_Pois = rng1.poisson(lam=N_Sig, size=1)[0]

    reference_npz = np.load(ref_filepath)
    if data_filepath == ref_filepath:
        data_npz = reference_npz
    else:
        data_npz = np.load(data_filepath)

    reference_all = reference_npz['data'][reference_npz['labels'] != anomaly_label]
    reference_labels_all = reference_npz['labels'][reference_npz['labels'] != anomaly_label]
    data_all = data_npz['data'][data_npz['labels'] != anomaly_label] if data_filepath != ref_filepath else reference_all
    data_labels_all = data_npz['labels'][data_npz['labels'] != anomaly_label] if data_filepath != ref_filepath else reference_labels_all
    anomaly_all = data_npz['data'][data_npz['labels'] == anomaly_label]
    anomaly_labels_all = data_npz['labels'][data_npz['labels'] == anomaly_label]

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

    anomaly_out = anomaly_all[idx_sig[:N_Sig_Pois]]
    anomaly_labels_out = anomaly_labels_all[idx_sig[:N_Sig_Pois]]
    data_out = data_all[idx_bkg[bkg_offset:bkg_offset+N_Bkg_Pois]]
    data_labels_out = data_labels_all[idx_bkg[bkg_offset:bkg_offset+N_Bkg_Pois]]
    reference_out = reference_all[idx_ref[:N_ref]]
    reference_labels_out = reference_labels_all[idx_ref[:N_ref]]

    return reference_out, reference_labels_out, data_out, data_labels_out, anomaly_out, anomaly_labels_out
