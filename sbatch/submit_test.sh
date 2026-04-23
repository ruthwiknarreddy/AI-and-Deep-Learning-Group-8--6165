#!/bin/bash 


cd ~/AI-and-Deep-Learning-Group-8--6165

module load anaconda3/2023.09
module load cuda
source activate deeplearning

python ./scripts/test_healthy_disease.py &> ./healthy_disease/logs/healthy_disease_test.2.log



python ./scripts/test_disease_type.py &> ./disease_type/logs/disease_type_test.2.log




