from src.myresnet import *
from src.mydata import *
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score,precision_score,accuracy_score
import torch

'''
Simple script to get the metrics on a particular model iteration
'''

data_path = "data/raw"

model_path = 'resnet18-2025-11-08_18:35/35.pth'

threshold = 0.8

model = ResnetNN()
model.load_state_dict(torch.load(f'models/{model_path}'))
stretch_resize = False
a_reg_transforms = get_albumentation_transforms(False, False)
test_set = XrayDataset(f"{data_path}/test", a_reg_transforms)
test_loader = DataLoader(test_set,128)
model.eval()
y_test = []
outputs_test = []
device = "cpu" if torch.cuda.is_available() else "cpu"
model.to(device)
for i,(x,y) in enumerate(test_loader):
    x,y = x.to(device),y.type(torch.FloatTensor).to(device)
    outputs = model(x)
    y_test.extend(y.cpu().tolist())
    outputs_test.extend(((F.sigmoid(outputs))>threshold).type(torch.int64).cpu().tolist())

recall = recall_score(y_test,outputs_test)
precision = precision_score(y_test,outputs_test)
accuracy = accuracy_score(y_test,outputs_test)
print(precision, recall,accuracy )
