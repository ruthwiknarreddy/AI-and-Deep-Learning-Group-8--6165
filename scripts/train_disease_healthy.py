from train_disease_healthy_utils import LoadDataset, show_img, plot_learning
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import os
import pandas as pd
import numpy as np

## set working directory
os.chdir(f"{os.path.expanduser('~')}/AI-and-Deep-Learning-Group-8--6165/")



if __name__ == "__main__":
    ### read in the label df ###
    label_df = pd.read_csv("./dataset/dataset_split.csv").loc[:,["files","label_binary"]]

    ## train_test_validation split, the dataset is unbalanced
    ## There are  15527 healthy samples.
    ## There are  43784 disease samples.
    ## approx. 20%-80% healthy-disease split
    ## Split instances into majority vs minority class/classes
    df_majority = label_df[label_df["label_binary"] == 'disease']
    df_minority = label_df[label_df["label_binary"] == 'healthy']

    # Undersampling majority class: so there is a 40%-60% healthy-disease split
    ## https://machinelearningmastery.com/navigating-imbalanced-datasets-with-pandas-and-scikit-learn/

    ## calculate number of desired majority class samples: Maj / (Maj+Min) = 0.6 --> round(.6*Min/.4) 
    df_majority_downsampled = df_majority.sample(n=int(.6/.4*len(df_minority)), random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    print(f"Original dataset: {len(label_df)}")
    print(f"Balanced dataset: {len(df_balanced)}") ## 38817 samples, 15527 from disease and 23290 from healthy


    train_df, temp_df = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced['label_binary'], random_state=0)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_binary'], random_state=0)


    ## resize images to 224x224 and rescale images to [0,1]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # automatically scales to [0,1]
    ])

    ### image generator
    train = LoadDataset(train_df, transform=transform)
    valid   = LoadDataset(val_df, transform=transform)
    test  = LoadDataset(test_df, transform=transform)



    #### plot the images ####
    plt.figure(figsize = (10,5))
    for i, sample_label in enumerate(train):
        if i < 10:
            plt.subplot(2,5,i+1)
            show_img(sample_label[0].permute(1, 2, 0).numpy(), sample_label[1])
        else:
            break
    plt.show()

    plt.tight_layout()
    plt.savefig("./healthy_disease/output/images/preliminary_data.png")



    #### augmentation pipeline ####
    ## flip horizontally and vertically, rotate, zoom. Brightness is adjusted to be 50% as bringht to 100% as bright. And adjust the contrast,saturation,hue between 0.9 and 1.1
    data_augmentation = v2.Compose([
        transforms.v2.RandomHorizontalFlip(p=0.5),
        transforms.v2.RandomVerticalFlip(p=0.5),
        transforms.v2.RandomRotation(degrees = (-.1,.1)),
        #transforms.v2.RandomZoomOut(p=.5),
        transforms.v2.ColorJitter(brightness = (.5,1), contrast = 0.1,
                                saturation = 0.1, hue = 0.1)
    ])



    #### see images after augmentation ####
    plt.figure(figsize=(10, 10))

    for i, image_label in enumerate(train):
        plt.figure(figsize=(10,8))
        plt.subplot(6, 4, 5)
        plt.imshow(image_label[0].permute(1, 2, 0).numpy())
        plt.axis("off")


        for j in range(9):
            ax = plt.subplot(3, 4, int(j/3)*4 + 2 + (j % 3))

            show_img(data_augmentation(image_label[0]).permute(1,2,0).numpy(), "")

            plt.tight_layout()
            plt.axis("off")
        plt.tight_layout()
        if i != 1:
            break
    plt.savefig("./healthy_disease/output/images/preliminary_augment_data.png")


    ######################################
    #### import the pretrained models ####
    ######################################

    retrain = True

    #### AlexNet ####


    #### GoogleNet aka InceptionNet V1. ####

