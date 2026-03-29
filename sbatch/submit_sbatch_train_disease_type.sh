#!/bin/bash

cd ~/AI-and-Deep-Learning-Group-8--6165/

if [ ! -d disease_type ]; then
	mkdir -p disease_type/output/train_test_results
	mkdir disease_type/output/images
	mkdir disease_type/models
	mkdir disease_type/logs
fi

sbatch ./sbatch/sbatch_train_disease_type.sbatch .8 ## dataset split: 0.2 train 0.4 valid 0.4 test
sbatch ./sbatch/sbatch_train_disease_type.sbatch .6 ## dataset split: 0.4 train 0.3 valid 0.3 test
sbatch ./sbatch/sbatch_train_disease_type.sbatch .5 ## dataset split: 0.5 train 0.25 valid 0.25 test
sbatch ./sbatch/sbatch_train_disease_type.sbatch .4 ## dataset split: 0.6 train 0.2 valid 0.2 test
sbatch ./sbatch/sbatch_train_disease_type.sbatch .2 ## dataset split: 0.8 train 0.1 valid 0.1 test
