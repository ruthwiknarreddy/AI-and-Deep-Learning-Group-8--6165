import streamlit as st
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
from io import StringIO
import os
import urllib.request
from huggingface_hub import hf_hub_download
import pickle


##########################
#### Preprocess image ####
##########################


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]  
transform = v2.Compose([
transforms.v2.Resize((224, 224)),
transforms.v2.ToImage(),
transforms.v2.ToDtype(torch.float32, scale=True),
transforms.v2.Normalize(mean, std)
])  

## Title
st.title("🌿 Plant Health Detector")
st.markdown("Upload a leaf image to check if it's healthy or diseased.")
st.divider()

## Take in images
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    ## display the image in the app
    

else: 
    exit("Please upload a jpg, jpeg, png.")

## transform data
image = transform(input_image).unsqueeze(0)

####################################
#### Predict Healthy vs Disease ####
####################################

## Since Alexnet with a 80/10/10 train/val/test split did best I will use this 

class HD_AlexNet(torch.nn.Module):
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

      
    def predict(self, data, labels = None):
        criterion = nn.BCEWithLogitsLoss()
        output_logits = self.model(data)#.squeeze()
        # print("logits shape: ",output_logits.shape)
        # print("labels shape: ",labels.shape)
        probs = torch.sigmoid(output_logits)
        if labels is not None:
            loss = criterion(output_logits, labels.float())
            return loss, (probs > 0.5).type(torch.int32)
        else:
            return probs, (probs > 0.5).type(torch.int32)
    
class DT_AlexNet(torch.nn.Module):
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

      
    def predict(self, data):
        criterion = nn.CrossEntropyLoss()
        output_logits = self.model(data)#.squeeze()
        # loss = criterion(output_logits, labels)
        return torch.softmax(output_logits, dim = 1), torch.argmax(output_logits, dim=1) ## largest logit is largest softmax


@st.cache_resource
def load_healthy_disease():
    healthy_disease = HD_AlexNet(retrain=True)

    healthy_disease_path = hf_hub_download(
     repo_id="Breannah/healthy_disease",
     filename="alexnet_model_healthy_disease.pt"
    )
    healthy_disease.model.load_state_dict(torch.load(healthy_disease_path, map_location=torch.device('cpu')))

    healthy_disease.model.eval() 
    
    return healthy_disease

healthy_disease = load_healthy_disease()

with torch.no_grad():
    HD_prob, predicted = healthy_disease.predict(image)



if  predicted.squeeze().item() == 0: ## Predict disease type
    
    # st.write(f"Your plant is diseased with a confidence of {(1 - prob.squeeze().item())*100:.2f}%. Predicting Disease Status.")
    @st.cache_resource
    def load_disease_type():
        disease_type = DT_AlexNet(retrain=True)

        disease_type_path = hf_hub_download(
        repo_id="Breannah/disease_type",
        filename="alexnet_model_disease_type.pt")


        disease_type.model.load_state_dict(torch.load(disease_type_path, map_location=torch.device('cpu')))

        disease_type.model.eval() 
        
        return disease_type
    
    disease_type = load_disease_type()

    with torch.no_grad():
        DT_prob, predicted = disease_type.predict(image)

    DT_prob = torch.max(DT_prob.squeeze()).item()

    disease_type_classes = hf_hub_download(
    repo_id="Breannah/disease_type",
    filename="disease_type_labels.pkl"
    )

    with open(disease_type_classes, "rb") as file:
            classes = pickle.load(file)
    

    classes = {i: disease for i, disease in enumerate(classes)}





########## APP ###########
st.sidebar.title("About")
st.sidebar.write("This model classifies plant disease.")




with col2:
    if uploaded_file:
        st.image(input_image, caption = "Your input image.")


st.divider()
st.subheader("Results")

if  predicted.squeeze().item() == 1: ## 1 means healthy
    st.write(f"Your plant is healthy with a confidence of {HD_prob.squeeze().item()*100:.2f}%")



else:
    plant, disease = classes[int(predicted.squeeze().item())].split(":")
    st.write(f"Your plant is diseased with a confidence of {(1 - HD_prob.squeeze().item())*100:.2f}%. Predicting Disease Status.")
    st.write( f"Your plant is {plant} afflicted by {disease[1:]} with a confidence of {DT_prob*100:.2f}%")



