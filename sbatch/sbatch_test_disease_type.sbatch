#!/bin/bash 
#SBATCH --partition=Nebula_GPU
#SBATCH --job-name=disease_type
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=80GB
#SBATCH --gres=gpu:4
#SBATCH --output=./disease_type/logs/disease_type_%j.log
#SBATCH --error=./disease_type/logs/disease_type_%j.log

echo "======================================================"
echo "Start Time : $(date)"
echo "Submit Dir : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Num Tasks : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo "" 
#cd $SLURM_SUBMIT_DIR
pwd

module load anaconda3/2023.09
module load cuda
source activate deeplearning


#### $1 is the size of the test and validation dataset combine
python ./scripts/test_disease_type.py 


mv ./disease_type/logs/disease_type_$SLURM_JOBID.log ./disease_type/logs/disease_type_test.2.log

echo ""
echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"


