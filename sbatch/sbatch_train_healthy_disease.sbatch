#!/bin/bash 
#SBATCH --partition=Nebula_GPU
#SBATCH --job-name=healthy_disease
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=80GB
#SBATCH --gres=gpu:4
#SBATCH --output=./healthy_disease/logs/healthy_disease_%j.log
#SBATCH --error=./healthy_disease/logs/healthy_disease_%j.log

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

c=$(echo "scale=2; $1/2" | bc)
tr=$(echo "scale=2; 1-$1" | bc)
echo "Train porportion of dataset: $tr"
echo "Validation porportion of dataset: $c"
echo "Test porportion of dataset: $c"


#### $1 is the size of the test and validation dataset combine
## so if $1 is .2 then the training size is .8, test size is .1 and valid is .1
python ./scripts/train_healthy_disease.py -ts $1


mv ./healthy_disease/logs/healthy_disease_$SLURM_JOBID.log ./healthy_disease/logs/healthy_disease_test_valid_size_$1.log

echo ""
echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"


