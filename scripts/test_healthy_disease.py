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
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
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
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["files"]
        label = self.df.iloc[idx]["label_binary"]
        if label == "healthy":
            label = 1
        else:
            label = 0

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label



class AlexNet(torch.nn.Module):
    def __init__(self, retrain = False):
        super().__init__()
        self.retrain = retrain
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        if retrain:
            ## Freeze bias and weights for all other layers
            for parameter in self.model.parameters():
                parameter.requires_grad = False


            self.model.classifier[-1] = nn.Linear(4096, 1)
            for layer in self.model.classifier[-3:]:
                for param in layer.parameters(): ## train the last two layers
                    param.requires_grad = True

  
    def forward(self, x):
        return self.model(x)

      
    def predict(self, data, labels):
        criterion = nn.BCEWithLogitsLoss()
        output_logits = self.model(data)#.squeeze()
        # print("logits shape: ",output_logits.shape)
        # print("labels shape: ",labels.shape)
        probs = torch.sigmoid(output_logits)
        loss = criterion(output_logits, labels.float())
        return loss, (probs > 0.5).type(torch.int32)
      
    
    


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
                        ('fc2', nn.Linear(500, 1))
                        ]))
            self.model.fc = fc_layers


    def forward(self, x):
        return self.model(x)

    
    def predict(self, data, labels):
        criterion = nn.BCEWithLogitsLoss()
        output_logits = self.model(data)#.squeeze()
        probs = torch.sigmoid(output_logits)
        loss = criterion(output_logits, labels.float())
        return loss, (probs > 0.5).type(torch.int32)


def test_model(test_data, num_samples, model_class, test_history_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_class.model.to(device)
    
    test_history = {"loss": [], "accuracy": [], "F1": [], "recall": [], "precision": []}

    accuracy = BinaryAccuracy().to(device)
    F1 = BinaryF1Score().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)

    batched_test = DataLoader(test_data, batch_size=num_samples, shuffle=False)

    
    print(f"                         Testing: {model_class.__class__.__name__}                         ")

    ####### validation #######
    model_class.model.eval() 

    with torch.no_grad(): ## Don't track gradients
        for batch in batched_test:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            labels = labels.int().unsqueeze(1)

           
            batch_loss, preds = model_class.predict(images, labels)
            test_loss = batch_loss.item()

            accuracy(preds, labels)
            F1(preds, labels)
            precision(preds, labels)
            recall(preds, labels)

                

    test_history["accuracy"].append(accuracy.compute().item())
    test_history["F1"].append(F1.compute().item())
    test_history["precision"].append(precision.compute().item())
    test_history["recall"].append(recall.compute().item())
    test_history["loss"].append(test_loss/len(batched_test))
    for key, value in test_history.items(): print(f"############################### {key}: {value[-1]} ###############################")
    print("\n\n")



    # torch.save(model_class.model.state_dict(), output_model_path)
    pd.DataFrame(test_history).to_csv(test_history_path)




    
     


if __name__ == "__main__":
    ### read in the label df ###
    label_df = pd.read_csv("./dataset/dataset_split.csv").loc[:,["files","label_binary"]]

    classes = {0:"disease", 1:"healthy"}

    #### troubleshooting dataset with fewer instances ########
    # throwaway, df_balanced = train_test_split(df_balanced, test_size=0.1, stratify=df_balanced['label_binary'], random_state=0)

    # train_df, temp_df = train_test_split(df_balanced, test_size=float(args.test_size), stratify=df_balanced['label_binary'], random_state=0)
    train_df, temp_df = train_test_split(label_df, test_size=0.2, stratify=label_df['label_binary'], random_state=0)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_binary'], random_state=0)



    

    #####################################
    #### train the pretrained models ####
    #####################################

    #################
    #### AlexNet ####
    #################

    #############################
    ##### Full augmentation #####
    #############################
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]    


    transform_valid = v2.Compose([
    transforms.v2.Resize((224, 224)),
    transforms.v2.ToImage(),
    transforms.v2.ToDtype(torch.float32, scale=True),
    transforms.v2.Normalize(mean, std)
    ])  

    ### image generator
    test = LoadDataset(test_df, transform=transform_valid)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet = AlexNet(retrain = True)
    alexnet.model = alexnet.model.to(device)
    alexnet.model.load_state_dict(torch.load("./healthy_disease/models/alexnet_model_test-size_.2.pt", map_location=torch.device(device)))




    test_model(test_data = test, num_samples = test_df.shape[0], model_class = alexnet, 
                test_history_path = f"./healthy_disease/output/train_test_results/alexnet_test_history_test-size_.2.csv")


    


    ########################################
    #### GoogLeNet aka InceptionNet V1. ####
    ########################################

    #############################
    ##### Full augmentation #####
    #############################



    transform_valid = v2.Compose([
    transforms.v2.Resize((224, 224)),
    transforms.v2.ToImage(),
    transforms.v2.ToDtype(torch.float32, scale=True),
    ])  

    ### image generator
    test = LoadDataset(test_df, transform=transform_valid)

    googlenet = GoogLeNet(retrain = True)
    googlenet.model = googlenet.model.to(device)
    googlenet.model.load_state_dict(torch.load("./healthy_disease/models/googlenet_model_test-size_.2.pt", map_location=torch.device(device)))

    criterion = nn.BCEWithLogitsLoss() ## loss

    
    test_model(test_data = test, num_samples = test_df.shape[0], model_class = googlenet, 
                test_history_path = f"./healthy_disease/output/train_test_results/googlenet_test_history_test-size_.2.csv")


