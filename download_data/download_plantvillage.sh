#!/bin/bash

cd ../dataset/PlantVillage

git clone https://github.com/spMohanty/PlantVillage-Dataset.git

mv PlantVillage-Dataset/raw/color/ .

rm -rf PlantVillage-Dataset