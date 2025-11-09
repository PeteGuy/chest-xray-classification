import torch


torch.manual_seed(0)
from torch.utils.data import dataset, dataloader
from torch.optim import adam, adamw
from torch.optim.lr_scheduler import  ReduceLROnPlateau,LRScheduler
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score
from src.myresnet import *
from src.mydata import *
from torchvision.models import resnet18,ResNet18_Weights

stretch_resize=False
use_pretrained=True
freeze_backbone = True
pretrained_head_layer_index = 60

def get_metrics(model, loader, device,loss_fn = None) -> tuple[float,float]:
    """
    Calculates the values of precision,recall and optionally loss for the dataset in the loader variable
    :param model: model to test
    :param loader: loader to use, should be either validation or test set
    :param device: cpu or gpu
    :param loss_fn: used to calculate loss, if undefined loss won't be calculated
    :return:
    """
    with torch.no_grad():
        output_val = []
        y_val = []
        for i, (x, y) in tqdm(enumerate(loader)):
            y = y.type(torch.FloatTensor).to(device)
            x = x.to(device)

            outputs = model(x)
            output_val.extend(((F.sigmoid(outputs))>0.8).type(torch.int32).cpu().tolist())
            y_val.extend((y).cpu().tolist())


            if (loss_fn):
                val_running_loss = loss_fn(outputs.squeeze(-1), y).item()*y.size()[0]
        loss = None
        if loss_fn:
            loss = val_running_loss / len(y_val)
        precision = precision_score(y_val, output_val)
        recall = recall_score(y_val, output_val)


    return precision, recall,loss


def train_step(
    model,
    optimizer,
    loss_fn,
    epoch,
    train_loader,
    val_loader,
    writer: SummaryWriter = None,
    device="cpu",
    scheduler: LRScheduler=None
) -> None:
    """
    Simple function to train one epoch
    :param model:
    :param optimizer:
    :param loss_fn:
    :param epoch:
    :param train_loader:
    :param val_loader:
    :param batch_size:
    :param writer:
    :param device:
    :param scheduler:
    :return:
    """
    model.train()
    running_loss = 0
    for i, (x, y) in tqdm(enumerate(train_loader)):
        y = y.type(torch.FloatTensor).to(device)
        x = x.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        loss = loss_fn(outputs.squeeze(-1), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*y.size()[0]

    model.eval()
    precision, recall,val_loss = get_metrics(model, val_loader, device,loss_fn)
    scheduler.step(val_loss)
    writer.add_scalars("metrics", {"recall": recall, "precision": precision}, epoch)
    writer.add_scalars("loss", {"loss_train": running_loss/len(train_set),"loss_val":val_loss}, epoch)


print("Setting up datasets")
print("Stretch resize is: ",stretch_resize)
print("Use pretrained: ",use_pretrained)


a_train_transforms = get_albumentation_transforms(True, stretch_resize)
a_reg_transforms = get_albumentation_transforms(False, stretch_resize)

#data_path = "../data/raw"
data_path = "data/raw"
train_set = XrayDataset(f"{data_path}/train", a_train_transforms)
val_set = XrayDataset(f"{data_path}/val", a_reg_transforms)
test_set = XrayDataset(f"{data_path}/test",a_reg_transforms)

############################# hyper parameters #################################

model = ResnetNN()
if(use_pretrained):
    imageresnet= resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for i,((name,param),(name_custom,param_custom)) in enumerate(zip(imageresnet.named_parameters(),model.named_parameters())):
        if(i >= pretrained_head_layer_index):
            break
        if param.requires_grad and param_custom.requires_grad:
            if(i == 0):
                param_custom.data = torch.mean(param.data,dim=1).unsqueeze(1)
            else:
                param_custom.data = param.data
    print("Weights have successfully been loaded")

if use_pretrained and freeze_backbone:
    model.freeze_backbone(verbose=False)

batch_size = 128
epochs = 40
learning_rate = 0.0005
patience = 4
loss_fn = nn.BCEWithLogitsLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device is : " + device)
optimizer = adamw.AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer,"min",patience=patience)

train_loader = dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = dataloader.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False)

model.to(device)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
print(current_time)

folder = f"./models/resnet18-{current_time}"
if not os.path.isdir(folder):
    os.makedirs(folder)
writer = SummaryWriter(f"runs/{current_time}")

save_freq = 4
for i in range(epochs):
    print("Epoch " + str(i))
    train_step(
        model,
        optimizer,
        loss_fn,
        i,
        train_loader,
        val_loader,
        writer=writer,
        device=device,
        scheduler=scheduler
    )
    if(i+1) % save_freq == 0:
        torch.save(model.state_dict(), f"{folder}/{i}.pth")


precision, recall,_ = get_metrics(model, test_loader, device)
print(precision)
print(recall)
print(f'Saving results at "./models/resnet18 - {current_time}.pth"')
print(f"Epochs saved in {folder}")
torch.save(model.state_dict(), f"./models/resnet18-{current_time}.pth")
writer.close()


