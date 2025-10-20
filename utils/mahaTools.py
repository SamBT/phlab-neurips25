import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .MAHALANOBISutils import compute_empirical_means,compute_empirical_cov_matrix,mahalanobis_test

def mahalanobis_dist(data, ref, ref_label,plot=True,fit=False,rule='sum'):#, sig_label=-1, seed=0, n_ref=1e4, n_bkg=1e3, n_sig=1e2, z_ratio=0.1, anomaly_type ='', plot=True, pois_ON=False):
    '''
    - computes the mahalnobis test for the dataset 
    '''
    # random seed                                                                                                                    
    #np.random.seed(seed)
    #print('Random seed: '+str(seed))
    
    # train on GPU?                                                                                                                  
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    #data   = data.to(DEVICE)
    #model  = model.to(DEVICE)
    #label  = label.to(DEVICE)

    # estimate parameters of the bkg model 
    means=compute_empirical_means(ref,ref_label)
    emp_cov=compute_empirical_cov_matrix(ref, ref_label, means)
    M_data = mahalanobis_test(data, means, emp_cov)
    if plot:
        M_ref  = mahalanobis_test(ref, means, emp_cov)
        # visualize mahalanobis
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        rMin=torch.min(M_ref)
        rMax=torch.max(M_ref)
        bins=np.linspace(rMin,rMax,20)
        plt.hist(M_ref,bins=bins,label='ref',alpha=0.5)
        plt.hist(M_data,bins=bins,label='data',alpha=0.5)
        #plt.hist([M_ref, M_data], density=True, label=['REF', 'DATA'])
        #font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, ncol=2, loc='best')
        #plt.yscale('log')
        #plt.yticks(fontsize=16, fontname='serif')
        #plt.xticks(fontsize=16, fontname='serif')
        plt.ylabel("density")#, fontsize=22, fontname='serif')
        plt.xlabel("mahalanobis metric")#, fontsize=22, fontname='serif')
        #plt.savefig(output_folder+'distribution.pdf')
        plt.show()
    if fit:
        M_ref  = mahalanobis_test(ref, means, emp_cov)
        result=fitDiff(-1.*M_data,-1.*M_ref)

    if rule=='sum':
        t = -1* torch.sum(M_data)
    elif rule=='max':
        t = -1* torch.min(M_data)
    #print('Mahalanobis test: ', "%f"%(t))
    return t,-1.*M_data

def fitDiff(data,ref):
    #start with binned fit to be easy
    rMin=torch.min(ref)
    rMax=torch.max(ref)
    bins=np.linspace(rMin,rMax,20)
    refhist,bin_edges  = np.histogram(ref, bins=bins)
    datahist,_         = np.histogram(data, bins=bins)
    x                  = 0.5*(bin_edges[1:] + bin_edges[:-1])
    tck                = interpolate.splrep(x, refhist)
    smodel             = lmfit.Model(gausSpline)
    bmodel             = lmfit.Model(spline)
    ps = smodel.make_params(mean=1,sigma=0.2,a1=100.0,a2=1.0)
    pb = bmodel.make_params(a2=1.)
    weights = 1./np.sqrt(np.maximum(refhist,0.1))
    resultb = bmodel.fit(data=datahist,params=pb,x=x,weights=weights,iTck=tck)
    lmfit.report_fit(resultb)
    results = smodel.fit(data=datahist,params=ps,x=x,weights=weights,iTck=tck)
    lmfit.report_fit(results)
    #plt.errorbar(x,datahist,yerr=np.sqrt(datahist),marker='o')
    #plt.errorbar(x,refhist*resultb.params['a2'].value,yerr=np.sqrt(refhist),marker='o')
    #plt.yscale('log')
    #plt.show()
    #return resultb
    #results.plot()
    #return resultsb.chisq-results.chisq
    return results