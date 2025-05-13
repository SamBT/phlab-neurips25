import os, json, glob, h5py, argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help="folder output", required=True)
args   = parser.parse_args()
folder = args.folder

dict_all = {}
header = ''
for file_tmp in glob.glob('%s/*.h5'%(folder)):
    print(file_tmp)
    if '_all' in file_tmp: continue
    f = h5py.File(file_tmp, 'r')
    print(file_tmp)
    if not len(list(dict_all.keys())):
        header = file_tmp.split('_')[-1].replace('.h5', '')
        for k in list(f.keys()):
            if "idx" in k: continue
            dict_all[k]= np.array([])

    for k in list(f.keys()):
        if "idx" in k: continue
        dict_all[k]=np.append(dict_all[k], np.array(f[k]))
    f.close()

f = h5py.File("%s/%s_all.h5"%(folder,header), 'w')
for k in list(dict_all.keys()):
    f.create_dataset(k, data=dict_all[k], compression='gzip')
f.close()
print('saved')
