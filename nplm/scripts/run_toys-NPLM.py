import os, json, argparse, glob, time, datetime
import numpy as np
import os.path

OUTPUT_DIRECTORY =  '../experiments_output/'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

# configuration dictionary                                                                                                \
config_json = {
    'ref_filepath': '',
    'data_filepath': '',
    'anomaly_label':-1,
    "N_Ref"   : 10000,
    "N_Bkg"   : 2000,                                                                                 
    "N_Sig"   : 20,
    "w_ref": 0.2,
    "output_directory": OUTPUT_DIRECTORY,
    "M" :100,
    "lambda": 1e-7,
    "iterations": 10000000,
    "falkon_sigmas": [0.1, 1.5, 2.6, 3.6, 4.9, 9.8], #[1, 25, 50, 75, 99, 99*2] perc of pair-wise distance                
    "Pois_ON": False
}                                                                      


#### launch python script ###########################                                                                                       \

if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-n','--name',    type=str, help="name of the experiment", required=True)
    parser.add_argument('-a','--anomaly-label', type=int, help="anomaly label", required=True)
    parser.add_argument('-r','--ref-filepath', type=str, help="reference file path", required=True)
    parser.add_argument('-d','--data-filepath', type=str, help="data file path", required=True)
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", default='toy-NPLM.py')
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default=1)
    parser.add_argument('-j', '--jobs',    type=int, help="number of jobs submissions",   required=False, default = 100)
    parser.add_argument('-s', '--nsig', type=int, help="number of signal events to inject", required=True)
    parser.add_argument('-o','--reference-only', action='store_true')
    parser.add_argument('--nref',type=int, help="number of reference events", required=False, default=10000)
    parser.add_argument('--nbkg',type=int, help="number of background events", required=False, default=2000)
    parser.add_argument('-q','--queue', type=str, help="queue to submit jobs to", required=False, default='iaifi_gpu_priority')
                                                                                    
    args     = parser.parse_args()
    ntoys    = args.toys
    njobs    = args.jobs
    pyscript = args.pyscript

    # set the number of signal events to inject
    config_json["N_Sig"] = args.nsig
    config_json["N_Ref"] = args.nref
    config_json["N_Bkg"] = args.nbkg
    config_json["anomaly_label"] = args.anomaly_label
    # set the reference file path
    config_json['ref_filepath'] = args.ref_filepath
    # set the data file path
    config_json['data_filepath'] = args.data_filepath
    # fix the number of kernels
    config_json['M'] = int(np.sqrt(args.nref + args.nbkg))

    # if we only want to run reference vs reference change file paths
    if args.reference_only:
        config_json['data_filepath'] = config_json['ref_filepath']

    # create the experiment label for saving
    # training specs                                                                                                   
    is_pois = ''
    if not config_json["Pois_ON"]:
        is_pois = 'NoPois'
    widths_str = ''
    for width in config_json["falkon_sigmas"]:
        widths_str += '%s-'%(str(np.around(width,1)))
    ID = f'/{args.name}/M{config_json["M"]}_lam{config_json["lambda"]}_iter{config_json["iterations"]}_sigma{widths_str}_{is_pois}/'
    # problem specs              
    ID += '/Nref'+str(config_json["N_Ref"])+'_Nbkg'+str(config_json["N_Bkg"])
    if config_json["N_Sig"]:
        ID += '_Nsig'+str(config_json["N_Sig"])
    if args.reference_only:
        ID += '_refOnly'

    # create output folder
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+ID
    if not os.path.exists(config_json["output_directory"]):                                                                   
        os.makedirs(config_json["output_directory"],exist_ok=True)


    config_json['pyscript'] = pyscript
    pyscript_str = pyscript.replace('.py', '')
    # save json file with problem settings
    json_path = create_config_file(config_json, config_json["output_directory"])  

    if args.local:
        #os.system("module load cuda/11.8.0-fasrc01")
        #os.system("module load python/3.10.9-fasrc01")
        os.system("python %s/%s -j %s -t %i"%(os.getcwd(), pyscript, json_path, ntoys))
    else:
        label = "logs"
        os.system("mkdir %s" %label)
        for i in range(njobs):
            script_sbatch = open("%s/submit_%i.sh" %(label, i) , 'w')
            script_sbatch.write("#!/bin/bash\n")
            script_sbatch.write("#SBATCH -c 1\n")
            script_sbatch.write("#SBATCH --gpus 1\n")
            script_sbatch.write("#SBATCH -t 0-3:15\n")
            script_sbatch.write(f"#SBATCH -p {args.queue}\n")
            script_sbatch.write("#SBATCH --mem=5000\n")
            script_sbatch.write("#SBATCH -o ./logs/%s"%(pyscript_str)+"_%j.out\n")
            script_sbatch.write("#SBATCH -e ./logs/%s"%(pyscript_str)+"_%j.err\n")
            script_sbatch.write("\n")
            #script_sbatch.write("module load python/3.10.9-fasrc01\n")
            #script_sbatch.write("module load cuda/11.8.0-fasrc01\n")
            script_sbatch.write("source ~/.bash_profile\n")
            script_sbatch.write("mamba activate torch_gpu\n")
            script_sbatch.write(f"cd {os.getcwd()}\n")
            script_sbatch.write("\n")
            script_sbatch.write("python %s -j %s -t %i\n"%(pyscript, json_path, ntoys))
            script_sbatch.close()
            os.system("chmod a+x %s/submit_%i.sh" %(label, i))
            os.system("sbatch %s/submit_%i.sh"%(label, i) )

