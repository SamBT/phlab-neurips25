import torch, json,os, time, datetime, umap
import FLKutils as flk
import GENutils as gen
import ANALYSISutils as an
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import mplhep as hep
hep.style.use("CMS")

def NPLM_routine(seed, json_path, calibration=False):
    '''
    generates a toy using seed.
    computes the mahalnobis test for the dataset.
    ARGS:
    - seed: random seed to extract the toy
    - json_path: path to json file containing the configuration distionary
    - calibration: boolean switching between signal injection and bkg-only toys.
                   if True, the toys are bkg-only
    RETURN:
    - t: the value of the NPLM test statistic for 6 different width choices. 
    - preds: the NPLM score for each event in the sample
    '''
    # random seed                                                                                                                    
    if seed==None:
        seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
    np.random.seed(seed)
    print('Random seed: '+str(seed))

    # setup parameters
    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)
    plot =  config_json["plot"]
    # problem definition                                                                                                             
    N_ref      = config_json["N_Ref"]
    N_Bkg      = config_json["N_Bkg"]
    N_Sig      = config_json["N_Sig"]
    if calibration: N_Sig=0
    luminosity_ratio    = config_json["luminosity_ratio"]
    Pois_ON    = config_json["Pois_ON"]
    anomalous_class = config_json["anomalous_class"]
    anomaly_type = config_json["anomaly_type"]
    sig_labels = [anomalous_class]
    #print('SIG classes: ', sig_labels)

    flk_M     = config_json["flk_M"]
    flk_lam   = config_json["flk_lambda"]
    flk_iter  = int(config_json["flk_iterations"])

    ##### define output path ######################                                                                                  
    OUTPUT_PATH    = config_json["output_directory"]
    OUTPUT_FILE_ID = '/seed%s/'%(seed)
    folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    output_folder = folder_out

    files = np.load(config_json['ref_path'])
    ref_all_x, ref_all_y = files['data'], files['labels']
    
    if calibration:
        # use the training sample
        files = np.load(config_json['ref_path'])
        data_all_x, data_all_y = files['data'], files['labels']
    else:
        # use the domain shifted sample
        files = np.load(config_json['data_path'])
        data_all_x, data_all_y = files['data'], files['labels']
    
    # build the dataset 
    ref_all_x = ref_all_x[ref_all_y!=anomalous_class]
    ref_all_y = ref_all_y[ref_all_y!=anomalous_class]
    sig_all_x = data_all_x[data_all_y==anomalous_class]
    sig_all_y = data_all_y[data_all_y==anomalous_class]
    bkg_all_x = data_all_x[data_all_y!=anomalous_class]
    bkg_all_y = data_all_y[data_all_y!=anomalous_class]
    
    # standardize                                                                                                                    
    mean_all, std_all = np.mean(ref_all_x, axis=0), np.std(ref_all_x, axis=0)
    std_all[std_all==0] = 1
    #print(mean_all, std_all)
    ref_all_x = gen.standardize(ref_all_x, mean_all, std_all)
    bkg_all_x = gen.standardize(bkg_all_x, mean_all, std_all)
    sig_all_x = gen.standardize(sig_all_x, mean_all, std_all)
    
    N_bkg_p = np.random.poisson(lam=N_Bkg, size=1)[0]
    N_sig_p = np.random.poisson(lam=N_Sig, size=1)[0]
    
    idx_bkg = np.arange(bkg_all_y.shape[0])
    np.random.shuffle(idx_bkg)
    idx_sig = np.arange(sig_all_y.shape[0])
    np.random.shuffle(idx_sig)
    
    feature = np.concatenate((sig_all_x[idx_sig[:N_sig_p]], bkg_all_x[idx_bkg[:N_bkg_p+N_ref]]), axis=0)
    target = np.concatenate((np.ones((N_sig_p+N_bkg_p, 1)), np.zeros((N_ref, 1))), axis=0)
    t, preds = NPLM_dist(feature, target, luminosity_ratio, flk_M=flk_M,
                         #flk_widths_perc=[1, 25, 50, 75, 99],
                         flk_lambda=flk_lam, flk_iterations=flk_iter)
    return t, preds, feature, target

def NPLM_dist(feature, target, luminosity_ratio, flk_M=None, flk_widths_perc=[1, 25, 50, 75, 99], flk_lambda=1e-7, flk_iterations=1e7):
    '''
    - data: experimental data, composed of bkg and signal
    - reference: bkg-only large sample of events that represents the null hypothesis
    - luminosity_ratio: N(bkg in data)/N(ref) 
                        in other words, is the proportion betweeen the experimental 
                        luminosity and the reference luminosity
    - flk_M: number of kernels in the falkon model used for NPLM
    - flk_widths_perc: values of kernel's width to be used for testing 
                       in percentiles of the typical pairwise-distance between points
    - flk_lambda: L2 regulariation coefficient applied to the kernels' coefficients amplitude
    - flk_iterations: max number of iterations the falkon algorithm is allowed to run for convergence
    RETURN:
    - numpy array of NPLM test outcome for different flk widths (shape: [Nwidths,])
    '''
    reference = feature[target[:, 0]==0]
    flk_widths = gen.candidate_sigma(reference[:2000, :], 
                                   perc=(flk_widths_perc))
    flk_widths= np.append(flk_widths, flk_widths[-1]*2) # very broad kernel
    print("width values: ", flk_widths)
    #feature = np.concatenate((data, ref), axis=0).astype(float)
    #target  = np.concatenate((data_y, ref_y), axis=0).astype(float)
    feature, target = feature.astype(float), target.astype(float) 
    cuda = torch.cuda.is_available()
    t_tmp = []
    preds_tmp = []
    for flk_sigma in flk_widths_perc:
        flk_seed = int(datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute)
        flk_config = flk.get_logflk_config(flk_M,flk_sigma,[flk_lambda],weight=luminosity_ratio,iter=[flk_iterations],seed=flk_seed,cpu=cuda)
        t, preds = flk.run_toy("", feature, target,  weight=luminosity_ratio, seed=flk_seed,
                              flk_config=flk_config, output_path='./', plot=False, verbose=False, df=10)
        t_tmp.append(t)
        preds_tmp.append(preds.reshape((-1, 1)))
    return np.array(t_tmp), np.concatenate(preds_tmp, axis=1) 

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
        p_ref, p_data =  an.min_p(t_null,t_alt)
    elif rule=='prod':
        p_ref, p_data = an.prod_p(t_null,t_alt)
    elif rule=='avg':
        p_ref, p_data = an.avg_p(t_null,t_alt)
    else:
        print("unrecognized rule. Available rules are : min, prod, avg.")
        return
    return p_ref, p_data

def plot_scores(seed, json_path, calibration=False):
    t, preds, feature, target = NPLM_routine(seed, json_path, calibration=calibration)
    data = feature[target[:, 0]==1]
    pred = preds[target[:, 0]==1]
    pred = torch.sigmoid(torch.from_numpy(pred)).numpy()
    print(np.min(pred), np.max(pred))
    
    plot_UMPA_score_colored(pred, data, seed, json_path, thr=0.5)
    plot_score_hist(preds, target, seed, json_path)
    return

def compute_statistics(json_path, Ntoys, output_path=None, power_thr=[1.645,2.33], rule='sum'):
    '''
    if a list of power thresholds are given, computes the power of the test at specified thresholds.
    computes the z-scores at 50%, 16%, 84% flase positive rate.
    ARGS:
    - json_path: path to json file containing the configuration distionary
    - Ntoys: number of toys to run for the test statistic distributions
    - rule: aggregation mode to compute the Mahalanobis test
    - power_thr: list of thresholds to compute the test power (results showed in the plot)
    - output_path: path to store figure. If None, the figure is not saved
    RETURN:
    - z_emp: numpy array of the z-score at 16%, 50%, 84% false negative rate (shape: [3,])
    '''
    if output_path==None: save=False
    else: save=True
    t_null, t_alt = [], []
    for seed in range(Ntoys):
        print('Calibration:')
        t_null.append(NPLM_routine(seed, json_path, calibration=True)[0])
        print('Alternative:')
        t_alt.append(NPLM_routine(seed, json_path, calibration=False)[0])
    t_null, t_alt = np.array(t_null), np.array(t_alt)
    t_null, t_alt = NPLM_combined_pval(t_null, t_alt, rule='min')
    if len(power_thr):
        # compute the power of the test at specified thresholds
        z_as, z_emp = an.plot_2distribution_new(t_null, t_alt, df=np.median(t_null), 
                                             xmin=np.min(t_null)-10, xmax=np.max(t_alt)+10, 
                                             nbins=8, save=save, output_path=output_path, 
                                             Z_print=power_thr,
                                             label1='REF', label2='DATA', 
                                             save_name='/combined-pvals', 
                                             print_Zscore=True)
    # compute z-scores at 50%, 16%, 84%
    z_emp = an.median_score(t_null,t_alt)[0]
    print("Z-score(FNR): %s (0.50), %s (0.16), %s (0.84)"%(str(np.around(z_emp[1], 2)), str(z_emp[0]), str(z_emp[2])))
    return z_emp

def NPLMLoss(true, pred, lumi_ratio):
    f   = pred[:, 0]
    y   = true[:, 0]
    return np.sum(lumi_ratio*(1-y)*(np.exp(f)-1) - y*(f))

def plot_UMPA_score_colored(preds, data, seed, json_path, thr=0.5):
    """
    plot a 2D UMAP of the data colored according to the score
    ARGS:
    - data: input to NPLM
    - preds: output of NPLM corresponding to the data (classifier score)
    - seed: the seed used to create the data (only used to store the plot)
    - thr: lower threshold applied to the score for plotting 
           (makes the scatter plot less dense and better readable)
    """
    print("UMAP plot")
    # setup output_folder                                                                                    
    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)
    OUTPUT_PATH    = config_json["output_directory"]
    OUTPUT_FILE_ID = '/seed%s/'%(seed)
    folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    # creat UMAP projection
    reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.1, random_state=42)
    reduced_data = reducer.fit_transform(data)
    
    for width in range(preds.shape[1]):
        print('width ', width, '...')
        # select according to threshold
        mask = preds[:, width]>thr
        color_tsne = preds[:, width]#:width+1]
        reduced_data_plot = reduced_data[mask]
        color_tsne = color_tsne[mask]#.reshape((-1,)) 
        # order in score
        order = np.argsort(color_tsne)
        reduced_data_plot=reduced_data_plot[order]
        color_tsne = color_tsne[order]
        # Create the scatter plot
        fig=plt.figure(figsize=(12, 10))
        ax =fig.add_axes([0.1, 0.1, 0.85, 0.85])
        fig.patch.set_facecolor('white')
        img=plt.scatter(reduced_data_plot[:, 0], reduced_data_plot[:, 1], c=color_tsne,
                        cmap='jet',#'PiYG_r',
                        s=60)
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        plt.title('UMAP')
        cbar=plt.colorbar(img)
        cbar.ax.set_ylabel('NPLM score')
        #plt.savefig(folder_out+'/UMAP_%s.pdf'%(str(width)))
        plt.savefig(folder_out+'/UMAP_%s.png'%(str(width)))
        plt.show()
        plt.close()
    return

def plot_score_hist(pred, label, seed, json_path):
    print("score histogram")
    # setup output_folder                                                                                             
    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)
    OUTPUT_PATH    = config_json["output_directory"]
    OUTPUT_FILE_ID = '/seed%s/'%(seed)
    folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    lumi_ratio = config_json["luminosity_ratio"]
    for width in range(pred.shape[1]):
        print('width ', width, '...')
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        bins = np.linspace(0., 1, 20)
        label=label.reshape((-1,))
        pred_i=pred[:, width]
        hD = plt.hist(pred_i[label==1],weights=np.ones(len(pred_i[label==1])), bins=bins,
                      label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(pred_i[label==0], weights=lumi_ratio*np.ones(len(pred_i[label==0])),
                      color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]),
                     color='black', ls='', marker='o', ms=5, zorder=3)
        font = font_manager.FontProperties(size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2, loc='best')
        plt.yticks(fontsize=16,)
        plt.xticks(fontsize=16,)
        plt.xlim(0, 1)
        plt.ylabel("events", fontsize=22)
        plt.xlabel("classifier output", fontsize=22)
        plt.yscale('log')
        plt.savefig(folder_out+'/score_%i.pdf'%(width))
        plt.show()
        plt.close()

    return
