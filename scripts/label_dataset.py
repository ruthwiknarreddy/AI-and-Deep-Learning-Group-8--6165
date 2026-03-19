#!/bin/bash


import pandas as pd
import os
import glob

os.chdir(f"{os.path.expanduser("~")}/AI-and-Deep-Learning-Group-8--6165/scripts")

## make the healthy vs disease split
def healthy_disease_split(folders: list, healthy_name: str):
    """
    folders are the globbed folders which contain healthy and disease folders
    healthy_name is the string which differentiates healthy from diseased plants
    """
    disease_files = []
    healthy_files = []
    disease_labels = [] ## for granular labels of the type of diseases

    for folder in folders:
        if healthy_name not in folder:
            disease_files += glob.glob(f"{folder}/*")
            disease_labels += [os.path.basename(folder).replace("___", ": ").replace("_", " ")]*len(glob.glob(f"{folder}/*"))

        else:
            healthy_files += glob.glob(f"{folder}/*")



    return disease_files, disease_labels, healthy_files


disease_files, disease_labels, healthy_files = healthy_disease_split(glob.glob("../dataset/PlantVillage/*"), "healthy")


temp_d, temp_dl, temp_h = healthy_disease_split(glob.glob("../dataset/pumpkin/*"), "Healthy")
disease_files += temp_d
disease_labels += temp_dl
healthy_files += temp_h


temp_d, temp_dl, temp_h = healthy_disease_split(glob.glob("../dataset/pear/Pear/leaves/*"), "healthy")
disease_files += temp_d
disease_labels += temp_dl
healthy_files += temp_h

print("There are ", len(healthy_files), "healthy samples.")
print("There are ", len(disease_files), "disease samples.") 
 
files_df = pd.concat([pd.DataFrame({"files": healthy_files, "label_binary": ["healthy"]*len(healthy_files)}),
    pd.DataFrame({"files": disease_files, "label_binary": ["disease"]*len(disease_files), "disease_label": disease_labels})])


files_df.to_csv("../dataset/dataset_split.csv", index=False)