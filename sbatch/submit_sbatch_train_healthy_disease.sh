#!/bin/bash

cd ~/AI-and-Deep-Learning-Group-8--6165

if [ ! -d healthy_disease ]; then
	mkdir -p healthy_disease/output/train_test_results
	mkdir healthy_disease/output/images
	mkdir healthy_disease/models
	mkdir healthy_disease/logs
fi

sbatch ./sbatch/sbatch_train_healthy_disease.sbatch .8 ## dataset split: 0.2 train 0.4 valid 0.4 test
sbatch ./sbatch/sbatch_train_healthy_disease.sbatch .6 ## dataset split: 0.4 train 0.3 valid 0.3 test
sbatch ./sbatch/sbatch_train_healthy_disease.sbatch .5 ## dataset split: 0.5 train 0.25 valid 0.25 test
sbatch ./sbatch/sbatch_train_healthy_disease.sbatch .4 ## dataset split: 0.6 train 0.2 valid 0.2 test
sbatch ./sbatch/sbatch_train_healthy_disease.sbatch .2 ## dataset split: 0.8 train 0.1 valid 0.1 test
