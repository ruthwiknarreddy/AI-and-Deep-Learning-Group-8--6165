#!/bin/bash 
#SBATCH --partition=Orion
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=800GB
#SBATCH --output=./healthy_disease/logs/healthy_disease_%j.log
#SBATCH --error=./healthy_disease/logs/healthy_disease_%j.log



cd ~/AI-and-Deep-Learning-Group-8--6165

module load anaconda3/2023.09
module load cuda
source activate deeplearning

python ./scripts/test_healthy_disease.py &> ./healthy_disease/logs/healthy_disease_test.log



python ./scripts/test_disease_type.py &> ./disease_type/logs/disease_type_test.log


rm ./healthy_disease/logs/healthy_disease_$SLURM_JOBID.log


echo ""
echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"
