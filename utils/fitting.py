import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import lmfit

def dLL(iData, iRef, iSig, iNSig, iNBkg, iNBins=100, xmin=0.0, xmax=1.0, plot=False,philBin=False,scale_data=1):
    #start with binned fit to be easy
    data_sort  = np.sort(iData)
    ntot   = len(data_sort)
    width  = int(ntot/iNBins)
    
    if philBin:
        #binsRange=[]
        #binsRange.append(1.1) # upper range
        #pVal=data_sort[-1]
        #for pBin in range(iNBins-1):
        #    pBinLow =  data_sort[-(width)*(pBin+1)]
        #    if pVal == pBinLow:
        #        continue
        #    binsRange.append(pBinLow)
        #    pVal=pBinLow
        #binsRange.append(-1)
        #bins=np.sort(np.array(binsRange))
        bins = np.percentile(iData, np.linspace(0, 100, iNBins+1))
    else:
        bins = np.linspace(xmin,xmax,iNBins+1)
    
    data,bin_edges = np.histogram(iData,bins=bins)
    data = data * scale_data
    ref_data,_  = np.histogram(iRef,bins=bins)
    sig_data,_  = np.histogram(iSig,bins=bins)
    
    ref_data  = ref_data * iNBkg/np.sum(ref_data)
    sig_data = sig_data * iNSig/np.sum(sig_data)
    
    x = (bins[:-1]+bins[1:])/2
    # fit splines to sig and bkg shape
    bkg_spline                = interpolate.splrep(x, ref_data)
    sig_spline                = interpolate.splrep(x, sig_data)
    
    def bkg_only_spline(x,a2):
        bkg = interpolate.splev(x, bkg_spline)*a2
        return bkg
        #binned = np.digitize(x,bins)-1
        #return a2*ref_data[binned]
    def sig_plus_bkg_spline(x,a1,a2):
        sig = interpolate.splev(x, sig_spline)*a1
        bkg = interpolate.splev(x, bkg_spline)*a2
        return sig+bkg
        #binned = np.digitize(x,bins)-1
        #return a1*sig_data[binned] + a2*ref_data[binned]
    
    #b_model             = lmfit.Model(bkg_only_spline)
    b_model = lmfit.Model(bkg_only_spline)
    params_b = b_model.make_params(a2=1.)
    
    #sb_model             = lmfit.Model(sig_plus_bkg_spline)
    sb_model = lmfit.Model(sig_plus_bkg_spline)
    params_sb = sb_model.make_params(a1=1.,a2=1.)
    
    weights = 1./np.sqrt(np.maximum(data,0.1))
    #weights = None
    #result_sb = sb_model.fit(data=data,params=params_sb,weights=weights,x=x,iTck1=sig_spline,iTck2=bkg_spline)
    result_sb = sb_model.fit(data=data,params=params_sb,weights=weights,x=x)
    #lmfit.report_fit(result_sb)
    if plot:
        result_sb.plot()
    #result_b  = b_model.fit(data=data,params=params_b,weights=weights,x=x,iTck=bkg_spline)
    result_b  = b_model.fit(data=data,params=params_b,weights=weights,x=x)
    #lmfit.report_fit(result_b)
    if plot:
        result_b.plot()
    #plt.errorbar(x,data,yerr=np.sqrt(datahist),marker='o')
    #plt.errorbar(x,ref_data*resultb.params['a2'].value,yerr=np.sqrt(refhist),marker='o')
    #plt.yscale('log')
    #plt.show()
    #return resultb
    #results.plot()
    return result_b.chisqr - result_sb.chisqr