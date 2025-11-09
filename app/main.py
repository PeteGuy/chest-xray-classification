import time

import albumentations.core.composition
import uvicorn
from typing import List,Tuple
import os
import base64
from src.myresnet import *
from src.mydata import *
from src.gradcam import get_attention_on_image
import torch
from fastapi import FastAPI,HTTPException
from fastapi import UploadFile,Body,File,Query,Request,BackgroundTasks
from fastapi.responses import FileResponse
from typing import Annotated
import sys
import logging
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchinfo
import uuid
# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='{asctime} - {name} - {levelname} - {message}',style='{')
# We want to log both in console and in a file
console_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler(filename="logging.txt")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load model and transforms
model = ResnetNN()
model.load_state_dict(torch.load('app/bestmodel.pth',map_location=torch.device('cpu')))
# Make don't use layers such as dropout
model.eval()
# Don't calculate gradient to reduce resource needs
torch.set_grad_enabled(False)
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)
a_transforms = get_albumentation_transforms(training=False, stretch=False)

old_stdout = sys.stdout # backup current stdout
sys.stdout = open(os.devnull, "w")
s = torchinfo.summary(model, (1, 1, input_size, input_size),device=device)
sys.stdout = old_stdout

logger.info("Loaded model, transformations and summary")

predict_batch_size = 64
threshold_score = 0.8

tmp_images = "tmp_images"
os.makedirs(tmp_images,exist_ok=True)

class FastAPI_dataset(Dataset):
    def __init__(self,files : List[Tuple[str,np.ndarray]],transforms):
        logger.debug(type(transforms))
        logger.debug(albumentations.core.composition.TransformType)
        self.files = files
        self.transforms = transforms

    def __getitem__(self, item):
        img = cv2.imdecode(self.files[item][1],cv2.IMREAD_GRAYSCALE)
        tensor = self.transforms(image=img)["image"]
        return self.files[item][0],tensor

    def __len__(self):
        return len(self.files)

def delete_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass



app = FastAPI()


@app.get("/")
async def hello():
    return {"message":"You are currently on the chest xray classification API !"}

@app.get("/metadata")
async def get_model_metadata(get_summary:Annotated[bool,Query(description="Determines whether the torchinfo summary which is rather large is sent in the response.")] = False):
    '''
    :param get_summary: whether to return a torchinfo summary of the model
    :return: Basic data about the model used
    '''
    result = {"Model":"ResNet-18",
            "Training specifications":"40 epochs, finetuning over image-net backbone, binary sigmoid classification over chest Xray images",
            "Metrics used":"Precision, recall"
            }
    if get_summary:
        result["Torchinfo summary"] = repr(s)
    return result

@app.get("/image/{image_name}")
async def get_image(image_name:str,background_tasks: BackgroundTasks):
    '''

    :param image_name: The name of the image inside of the tmp_images folder
    :param background_tasks: used to delete the image after it's used once
    :return: an image
    '''
    file_path = os.path.join(tmp_images,image_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404,detail="Image not found, note that images get deleted after being seen once.")


    background_tasks.add_task(delete_file,file_path)
    return FileResponse(file_path,media_type="image/jpeg")



@app.post("/predict/")
async def predict(request: Request,file: UploadFile, show_attention:Annotated[bool,Query(description="If this is true, we return an image showing the attention of the model using the gradcam method.")]=False):
    '''
    :param request:
    :param file: the image to be classified, should be a png or jpeg image of the xray of a torso
    :param show_attention: whether to send back url to the image after gradcam was applied to it
    :return: the sigmoid score of the model, the class and potentially a url
    '''
    if file.content_type.strip() != "image/jpeg" and file.content_type.strip() != "image/png":
        raise HTTPException(status_code=422,detail="Invalid content type, file must be an image of type jpeg or png")
    img_str = await file.read()
    np_arr = np.frombuffer(img_str,np.uint8)
    logger.debug(file.content_type)
    img = cv2.imdecode(np_arr,cv2.IMREAD_GRAYSCALE)
    augmented = a_transforms(image=img)["image"]
    augmented = augmented.unsqueeze(dim=0)
    augmented = augmented.to(device)
    proba = model.predict_probas(augmented)

    class_n = (proba> threshold_score).type(torch.bool).cpu().item()
    if class_n:
        classe= "PNEUMONIA"
    else:
        classe = "NORMAL"
    result = {"Score":proba.item(),"Class":classe}
    if show_attention:
        with torch.set_grad_enabled(True):
            img_name = uuid.uuid4()
            full_path = os.path.join(tmp_images,f"{img_name}.jpg")
            cv2.imwrite(full_path,img=cv2.cvtColor(get_attention_on_image(model,augmented),cv2.COLOR_RGB2BGR))
            result["attention_url"] = f"{request.base_url}image/{img_name}.jpg"

    return result


@app.post("/predict_batch/")
async def predict(files: list[UploadFile]):
    '''
    :param files:
    :return: a list where each entry contains a filename, a class and a score for the corresponding image
    '''
    # Turn the files of type jpeg or png into arrays to be processed, ignore the incorrect types
    files_array = [(f.filename,np.frombuffer(await f.read(),np.uint8)) for f in files if f.content_type.strip() == "image/jpeg" or f.content_type.strip() == "image/png"]
    if len(files_array) == 0:
        logger.error("Invalid content type, none of the submitted files are images of type jpeg or png.")
        raise HTTPException(status_code=422,detail="Invalid content type, none of the submitted files are images of type jpeg or png.")

    result = {}
    if len(files_array) != len(files):
        logger.warning("One or more of the files uploaded were of invalid types, only correct files will be returned")
        result["message"] ="One or more of the files uploaded were of invalid types, only correct files were processed"
    # Use a dataset for iteration over many images
    dataset = FastAPI_dataset(files_array,a_transforms)
    loader = DataLoader(dataset,shuffle=False,batch_size=predict_batch_size)
    results = []
    for names,x in loader:
        x = x.to(device)
        probas = model.predict_probas(x).squeeze()
        # Reshape handles the case where there is only one result
        prob_list = probas.cpu().reshape(-1).tolist()
        logger.debug(prob_list)
        results.extend([{"name":names[i],"score":p,"class":"PNEUMONIA" if p>threshold_score else "NORMAL"} for i,p in enumerate(prob_list)])

    result["results"] = results
    return result



if __name__ == "__main__":
    uvicorn.run("app.main:app",host="0.0.0.0",port=8000,reload=True)