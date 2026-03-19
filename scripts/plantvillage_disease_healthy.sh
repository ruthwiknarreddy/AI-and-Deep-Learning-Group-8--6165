#!/bin/bash


cd ../dataset/
#### PlantVillage ####
cp -r `find ./PlantVillage/ -mindepth 1 -type d ! -path '*healthy*' ` ./disease
## handle special characters
find ./PlantVillage/ -mindepth 1 -type d ! -path '*healthy*' -print0 | xargs -0 cp -r -t ./disease

## amount of diseased plants from plant village
find ./PlantVillage/ -type f ! -path "*healthy*" | wc -l
39221