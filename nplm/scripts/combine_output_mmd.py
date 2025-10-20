import os, json, glob, h5py, argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help="folder output", required=True)
args   = parser.parse_args()
folder = args.folder

output = []
header = ''
for file_tmp in glob.glob('%s/*.npy'%(folder)):
    print(file_tmp)
    if '_all' in file_tmp: continue
    f = np.load(file_tmp)
    header = file_tmp.split('_')[-1].replace('.npy', '')
    output.append(f)
output = np.concatenate(output, axis=0)
np.save("%s/%s_all.npy"%(folder,header), output)