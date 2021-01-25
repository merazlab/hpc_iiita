#!/bin/bash

runfile="pytorch_gpu.py"
env_name="mptorch17_102"
#Make executable chmod +x task.sh
#run this way only . ./task.sh
########################################
cwd=$(pwd)
date1=$(date +"%s")

echo ---------------------------------------------
echo ----------------JOB Details------------------
echo ---------------------------------------------


echo Job Start Time:-- `date`
echo Working directory is $cwd

conda activate $env_name
echo "$env_name is now active "

#Change to proj working dir
cd $cwd

now=$(date +"%Y%m%d_%H%M%S")
log_name="out_"$now""

#intractive mode job run
python3 -u $runfile

#NonIntractive mode
# nohup python3 -u $runfile > $log_name.log & 


conda deactivate
echo --------------------------------------------
echo --------------Job Summery-------------------
echo --------------------------------------------

date2=$(date +"%s")

duration=$(($date2 - $date1))
seconds=$((duration % 60))
min=$((duration / 60))
minutes=$((min % 60))
hours=$((min / 60))

echo Job Duration:- $hours H : $minutes M : $seconds S

