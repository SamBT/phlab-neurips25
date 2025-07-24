import numpy as np
import os, time
import torch

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
font = font_manager.FontProperties(family='serif', size=20)


def get_logflk_config(M,flk_sigma,lam,weight,iter=[1000000],seed=None,cpu=False):
    # it returns logfalkon parameters
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,
            'penalty_list' : lam, # list of regularization parameters,
            'iter_list' : iter, #list of number of CG iterations,
            'options' : FalkonOptions(cg_tolerance=np.sqrt(float(1e-7)), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # (int or None), the model seed (used for Nystrom center selection) is manually set,
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }

def compute_t(preds,Y,weight):
    # it returns extended log likelihood ratio from predictions
    diff = weight*np.sum(1 - np.exp(preds[Y==0]))
    return 2 * (diff + np.sum(preds[Y==1]))

def trainer(X,Y,flk_config):
    # trainer for logfalkon model
    Xtorch=torch.from_numpy(X)
    Ytorch=torch.from_numpy(Y)
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model.predict(Xtorch).numpy()

def return_best_chi2dof(tobs):
    """
    Returns the most fitting value for dof assuming tobs follows a chi2_dof distribution,
    computed with a Kolmogorov-Smirnov test, removing NANs and negative values.
    Parameters
    ----------
    tobs : np.ndarray
        observations
    Returns
    -------
        best : tuple
            tuple with best dof and corresponding chi2 test result
    """
    dof_range = np.arange(np.nanmedian(tobs) - 10, np.nanmedian(tobs) + 10, 0.1)
    ks_tests = []
    for dof in dof_range:
        test = kstest(tobs, lambda x:chi2.cdf(x, df=dof))[0]
        ks_tests.append((dof, test))
    ks_tests = [test for test in ks_tests if test[1] != 'nan'] # remove nans
    ks_tests = [test for test in ks_tests if test[0] >= 0] # retain only positive dof
    best = min(ks_tests, key = lambda t: t[1]) # select best dof according to KS test result

    return best

def emp_zscore(t0,t1):
    if max(t0) <= t1:
        p_obs = 1 / len(t0)
        Z_obs = round(norm.ppf(1 - p_obs),2)
        return Z_obs
    else:
        p_obs = np.count_nonzero(t0 >= t1) / len(t0)
        Z_obs = round(norm.ppf(1 - p_obs),2)
        return Z_obs

def chi2_zscore(t1, dof):
    p = chi2.cdf(float('inf'),dof)-chi2.cdf(t1,dof)
    return norm.ppf(1 - p)


def run_toy(test_label, X_train, Y_train, weight, flk_config, seed, plot=False, verbose=False, savefig=False, output_path='', df=10):
    '''
    type of signal: "NP0", "NP1", "NP2", "NP3"
    output_path: directory (inside ./runs/) where to save results
    N_0: size of ref sample
    N0: expected num of bkg events
    NS: expected num of signal events
    flk_config: dictionary of logfalkon parameters
    toys: numpy array with seeds for toy generation
    plots_freq: how often to plot inputs with learned reconstructions
    df: degree of freedom of chi^2 for plots
    '''
    if not os.path.exists(output_path):
      os.makedirs(output_path, exist_ok=True)
    #save config file
    with open(output_path+"/flk_config.txt","w") as f:
        f.write( str(flk_config) )
    dim = X_train.shape[1]
    # learn_t
    flk_config['seed']=seed # select different centers for different toys
    st_time = time.time()
    preds = trainer(X_train,Y_train,flk_config)
    t = compute_t(preds,Y_train,weight)
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---LRT = {}\n---Time = {} sec\n\t".format(seed,t,dt))
    #with open(output_path+"t.txt", 'a') as f:
    #    f.write('{},{}\n'.format(seed,t))
    if plot:
        plot_reconstruction(data=X_train[Y_train.flatten()==1], weight_data=1, ref=X_train[Y_train.flatten()==0], weight_ref=weight,
                            df=df, t_obs=t, ref_preds=preds[Y_train.flatten()==0],
                            save=savefig, save_path=output_path+'/plots/', file_name=test_label+'.pdf'
                )
    return t, preds



def plot_reconstruction(data, weight_data, ref, weight_ref, ref_preds, xlabels=[], yrange=None, binsrange=None,
                        save=False, save_path='', file_name=''):
    '''             
    Reconstruction of the data distribution learnt by the model.  
    df:              (int) chi2 degrees of freedom     
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)   
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)  
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)  
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample   
    tau_OBS:         (float) value of the tau term after training          
    output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training 
    feature_labels:  (list of string) list of names of the training variables          
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels) 
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)   
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable 
    (ymax_code.keys()=feature_labels)                  
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted)   
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training 
    (if not given, only tau reconstruction is plotted)
    '''
    # used to regularize empty reference bins                                                                                                                           
    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data

    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
        bins = np.linspace(np.min(ref[:, i]),np.max(ref[:, i]),50)
        if not binsrange==None:
            if len(binsrange[xlabels[i]]):
                bins=binsrange[xlabels[i]]
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax1= fig.add_axes([0.1, 0.43, 0.8, 0.5])
        hD = plt.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(ref[:, i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = plt.hist(ref[:, i], weights=np.exp(ref_preds[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)

        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label='RECO', color='#b2df8a', lw=1, s=30, zorder=4)

        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3])
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/(hR[0]+eps), yerr=np.sqrt(hD[0])/(hR[0]+eps), ls='', marker='o', label ='DATA/REF', color='black')
        plt.plot(x, hN[0]/(hR[0]+eps), label ='RECO', color='#b2df8a', lw=3)
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        
        if xlabels:
            plt.xlabel(xlabels[i], fontsize=22, fontname='serif')
        else:
            plt.xlabel('x', fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')

        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        #plt.ylim(0,10)
        if len(xlabels):
            if not yrange==None and len(xlabels)>0:
                plt.ylim(yrange[xlabels[i]][0], yrange[xlabels[i]][1]) 
        plt.grid()
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path+file_name)
        plt.show()
        plt.close()
    return
