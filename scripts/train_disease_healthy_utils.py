#!/bin/bash

######################################
######    Healthy vs Disease    ######
######################################



from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
# import numpy as np
import os
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
from PIL import Image
## set working directory
os.chdir(f"{os.path.expanduser('~')}/AI-and-Deep-Learning-Group-8--6165/")


class LoadDataset(Dataset):
    """Load dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["files"]
        label = self.df.iloc[idx]["label_binary"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# print(classes)
## print images
def show_img(x, title="", bot=""): ## bot is bottom
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.xlabel(bot)

def plot_learning(history, title, label: str, metric, val_metric, output_dir = None):
    """
      title: graph name
      label: y-axis label
      metric: metric you're plotting
      val_metric: validation metric you're plotting
    """
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history[val_metric], label=val_metric)
    plt.xlabel('Epoch')
    plt.ylabel(f'{label}')
    plt.title(title)
    plt.legend()

    if output_dir:
      plt.savefig(output_dir)
    else:
      plt.show() 



def split_dataset(df):
  # TODO
  ...


