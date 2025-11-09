import cv2
from pytorch_grad_cam import GradCAM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.myresnet import *
from torchinfo import summary
from src.mydata import *
import torch
import numpy as np

# Input should be an image ready to be processed by our model
def get_attention_on_image(model:nn.Module,input:torch.Tensor):
    target_layers = [dict(model.named_modules())["layer4.1.c2"]]
    targets = None
    cam = GradCAM(model, target_layers)
    grayscale_cam = cam(input_tensor=input, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    final = cv2.cvtColor(np.asarray(input.squeeze(0).squeeze(0).cpu()), cv2.COLOR_GRAY2RGB)
    visualization = show_cam_on_image(final, grayscale_cam, use_rgb=True)
    return visualization
if __name__ == "__main__":

    model = ResnetNN()
    model.load_state_dict(torch.load(f"./models/resnet18-2025-10-28_11:38/35.pth"))
    model.eval()
    target_layers = [dict(model.named_modules())["layer4.1.c2"]]
    stretch_resize = False

    a_reg_transforms = getAlbumentationTransforms(False,stretch_resize)

    img_path = "data/raw/test/PNEUMONIA/person1_virus_7.jpeg"
    #img_path = "./person1_virus_6.jpeg"
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    input_tensor = a_reg_transforms(image=img)['image'].unsqueeze(0)

    targets = None
    cam = GradCAM(model,target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    final = cv2.cvtColor(np.asarray(input_tensor.squeeze(0).squeeze(0)),cv2.COLOR_GRAY2BGR)

    visualization = show_cam_on_image(final, grayscale_cam, use_rgb=False)
    visualization = cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./gradcamcv.png',cv2.cvtColor(visualization,cv2.COLOR_RGB2BGR))