to run the test:
python run_toys-NPLM.py -p toy-NPLM.py -t 100 -j 1

arguments:
-t (--ntoys): number of toys per job
-j (--njobs): number of jobs to send on cannon
-l (--local): if you want to run a job locally
-p (--pyscript): python script with test routing defined

output:
one temporary .h5 file from each cannon job.

to combine them in a single file run:
python combine_tmp_output.py -f [folder-path-where-outputs-are-saved]

the notebook contains the code to analyse and plot the test
