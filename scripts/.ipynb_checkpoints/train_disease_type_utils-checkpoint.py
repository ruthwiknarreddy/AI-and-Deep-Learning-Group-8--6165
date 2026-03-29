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
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)
from torchvision.transforms import v2

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
        self.classes = {disease: i for i, disease in enumerate(df["disease_label"].unique())}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["files"]
        label = self.df.iloc[idx]["disease_label"]
        label = self.classes[label]

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



class AlexNet(torch.nn.Module):
    def __init__(self, retrain = False):
        super().__init__()
        self.retrain = retrain
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        if retrain:
            ## Freeze bias and weights for all other layers
            for parameter in self.model.parameters():
                parameter.requires_grad = False


            self.model.classifier[-1] = nn.Linear(4096, 33)
            for layer in self.model.classifier[-3:]:
                for param in layer.parameters(): ## train the last two layers
                    param.requires_grad = True

  
    def forward(self, x):
        return self.model(x)

      
    def predict(self, data, labels):
        criterion = nn.CrossEntropyLoss()
        output_logits = self.model(data)#.squeeze()
        loss = criterion(output_logits, labels)
        return loss, torch.argmax(output_logits, dim=1) ## largest logit is largest softmax
      
    
    


class GoogLeNet(torch.nn.Module):
    def __init__(self, retrain = False):
        super().__init__()
        self.retrain = retrain
        self.model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        if retrain:
            ## Freeze bias and weights for all other layers
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            from collections import OrderedDict
            fc_layers = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(1024, 500)),
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(500, 33))
                        ]))
            self.model.fc = fc_layers


    def forward(self, x):
        return self.model(x)

    
    def predict(self, data, labels):
        criterion = nn.CrossEntropyLoss()
        output_logits = self.model(data)#.squeeze()
        loss = criterion(output_logits, labels)
        return loss, torch.argmax(output_logits, dim=1)


def train_model(train_data, valid_data, model_class, optimizer, criterion,
                output_model_path: str, train_history_path: str, valid_history_path: str, epochs = 2):
    
    data_augmentation = v2.Compose([
    transforms.v2.RandomHorizontalFlip(p=0.5),
    transforms.v2.RandomVerticalFlip(p=0.5),
    transforms.v2.RandomRotation(degrees = (-.1,.1)),
    #transforms.v2.RandomZoomOut(p=.5),
    transforms.v2.ColorJitter(brightness = (.5,1), contrast = 0.1,
                            saturation = 0.1, hue = 0.1)
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_class.model.to(device)

    num_classes = 33
    
    train_history = {"loss": [], "accuracy": [], "F1": [], "recall": [], "precision": []}
    valid_history = {"loss": [], "accuracy": [], "F1": [], "recall": [], "precision": []}

    accuracy = MulticlassAccuracy(num_classes=num_classes, average = "weighted").to(device)
    F1 = MulticlassF1Score(num_classes=num_classes, average = "weighted").to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average = "weighted").to(device)
    recall = MulticlassRecall(num_classes=num_classes, average = "weighted").to(device)

    batched_train = DataLoader(train_data, batch_size=64, shuffle=True)
    batched_valid = DataLoader(valid_data, batch_size=64, shuffle=False)

    
    print(f"                         Training {model_class.__class__.__name__}                         ")
    ## same number of epochs in https://doi.org/10.3389/fpls.2016.01419
    for EPOCH in range(epochs):  ## 30 epochs
        train_loss = 0
        model_class.model.train()
        ####### training #######
        for batch in batched_train:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            images = data_augmentation(images)
            
            # print(images.shape, labels.shape)

            # 1. Forward pass: Compute predicted y by passing inputs to the model
            output_logits = model_class.model(images)#.squeeze() ## images
            loss = criterion(output_logits, labels)
            train_loss += loss.item() ## loss for each epoch

            # 2. Zero the parameter gradients
            optimizer.zero_grad() ## remove past parameter gradients

            # 3. Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()

            # 4. Optimizer: Update parameters
            optimizer.step()

            ## make predictions and compute metrics
            probs = torch.softmax(output_logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            accuracy(preds, labels)
            F1(preds, labels)
            precision(preds, labels)
            recall(preds, labels)

        train_history["accuracy"].append(accuracy.compute().item())
        train_history["F1"].append(F1.compute().item())
        train_history["precision"].append(precision.compute().item())
        train_history["recall"].append(recall.compute().item())
        train_history["loss"].append(train_loss/len(batched_train))

        #### report training ####
        print(f"############################### Training Epoch: {EPOCH} of {epochs} done ###############################")
        for key, value in train_history.items(): print(f"############################### {key}: {value[-1]} ###############################")


        accuracy.reset() ; F1.reset() ; precision.reset() ; recall.reset()
        ####### validation #######
        model_class.model.eval() 
        valid_loss = 0
        with torch.no_grad(): ## Don't track gradients
            for batch in batched_valid:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                labels = labels.long()

                # print(images.shape, labels.shape)

                ## predict validation data after each epoch
                
                batch_loss, preds = model_class.predict(images, labels)
                valid_loss += batch_loss.item()

                accuracy(preds, labels)
                F1(preds, labels)
                precision(preds, labels)
                recall(preds, labels)

                

            valid_history["accuracy"].append(accuracy.compute().item())
            valid_history["F1"].append(F1.compute().item())
            valid_history["precision"].append(precision.compute().item())
            valid_history["recall"].append(recall.compute().item())
            valid_history["loss"].append(valid_loss/len(batched_valid))
            print(f"\n############################### Valid Epoch: {EPOCH} of {epochs} done ###############################")
            for key, value in valid_history.items(): print(f"############################### {key}: {value[-1]} ###############################")
            print("\n\n")

    torch.save(model_class.model.state_dict(), output_model_path)
    pd.DataFrame(train_history).to_csv(train_history_path)
    pd.DataFrame(valid_history).to_csv(valid_history_path)




    
     

