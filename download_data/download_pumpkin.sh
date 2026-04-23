#!/bin/bash

base_dir=~/AI-and-Deep-Learning-Group-8--6165/dataset

curl -L -o $base_dir/pumpkin-leaf-diseases-dataset-from-bangladesh.zip \
  https://www.kaggle.com/api/v1/datasets/download/tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh


unzip $base_dir/pumpkin-leaf-diseases-dataset-from-bangladesh.zip \
	"Pumpkin Leaf Diseases Dataset From Bangladesh/Original Dataset/*" \
	-d $base_dir/pumpkin

mv $base_dir/pumpkin/"Pumpkin Leaf Diseases Dataset From Bangladesh/Original Dataset/"* $base_dir/pumpkin

rm -rf $base_dir/pumpkin/"Pumpkin Leaf Diseases Dataset From Bangladesh/"



