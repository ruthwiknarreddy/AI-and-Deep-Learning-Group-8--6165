import requests
import os
import zipfile

url = f"https://zenodo.org/api/records/5557313"

# Get metadata
r = requests.get(url)
data = r.json()

# Loop through files
for f in data["files"]:
    filename = f["key"]
    
    # Filter images
    # if filename.lower().endswith((".png", ".jpg", ".jpeg")):
    download_url = f["links"]["self"]
        
    print("Downloading:", filename)
    file_data = requests.get(download_url)
        
    os.makedirs("images", exist_ok=True)
    with open(os.path.join("../dataset/pear", filename.split("/")[-1]), "wb") as out:
        out.write(file_data.content)
