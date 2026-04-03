#!/bin/bash
#SBATCH --partition=Nebula_GPU
#SBATCH --job-name=gradcam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=../healthy_disease/logs/gradcam_%j.log
#SBATCH --error=../healthy_disease/logs/gradcam_%j.log

echo "======================================================"
echo "Start Time  : $(date)"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "======================================================"

cd ../
pwd

module load anaconda3/2023.09
module load cuda
source activate deeplearning

echo "Test size: $1"

python ./scripts/run_gradcam_cluster.py -ts $1

mv ./healthy_disease/logs/gradcam_$SLURM_JOBID.log ./healthy_disease/logs/gradcam_test-size_$1.log

echo ""
echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"
