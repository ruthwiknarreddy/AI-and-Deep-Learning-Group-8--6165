#!/bin/bash

######################################
######    Healthy vs Disease    ######
######################################



from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print("TF version:", tf.__version__)

# import tensorflow as tf

import matplotlib.pyplot as plt
# import numpy as np

# import keras_tuner

import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'

import pandas as pd


# from keras.utils import plot_model

## set working directory
os.chdir(f"{os.path.expanduser("~")}/AI-and-Deep-Learning-Group-8--6165/scripts")

### read in the label df ###
label_df = pd.read_csv("../dataset/dataset_split.csv").loc[:,["files","label_binary"]]

## train_test_validation split, the dataset is unbalanced
## There are  15527 healthy samples.
## There are  43784 disease samples.
## approx. 20%-80% healthy-disease split
## Split instances into majority vs minority class/classes
df_majority = label_df[label_df["label_binary"] == 'disease']
df_minority = label_df[label_df["label_binary"] == 'healthy']

# Undersampling majority class: so there is a 40%-60% healthy-disease split
## https://machinelearningmastery.com/navigating-imbalanced-datasets-with-pandas-and-scikit-learn/

## calculate number of desired majority class samples 
## Maj / (Maj+Min) = 0.6 --> round(.6*Min/.4) 
df_majority_downsampled = df_majority.sample(n=int(.6/.4*len(df_minority)), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

print(f"Original dataset: {len(label_df)}")
print(f"Balanced dataset: {len(df_balanced)}") ## 38817 samples, 15527 from disease and 23290 from healthy

## 70 / 15 / 15 train / validation / test split
train_df, temp_df = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced['label_binary'], random_state=0)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_binary'], random_state=0)


### image generator
imagegen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

## train
train = imagegen.flow_from_dataframe(train_df, x_col = "files", y_col="label_binary", target_size=(224,224), batch_size = 128, shuffle = True)
## test
test = imagegen.flow_from_dataframe(test_df, x_col = "files", y_col="label_binary", target_size=(224,224), batch_size = 128, shuffle = True)
## validation
valid = imagegen.flow_from_dataframe(val_df, x_col = "files", y_col="label_binary", target_size=(224,224), batch_size = 128, shuffle = True)




## print images
def show_img(x, title="", bot=""): ## bot is bottom
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.xlabel(bot)

images, labels = next(train)

plt.figure(figsize = (10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    show_img(images[i], labels[i])

plt.tight_layout()
plt.savefig("../name.png")







