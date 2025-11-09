from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.transforms as transforms
import os
import albumentations as A
import cv2
from src.myresnet import input_size

mean = 0
std = 1
class XrayDataset(Dataset):
    def __init__(self, folderpath: str, transform=A.ToTensorV2()) -> None:
        # Load every names and corresponding label
        # We assume the folderpath will always contain a PNEUMONIA and a NORMAL folder
        images_paths = [
            os.path.join(folderpath, f"PNEUMONIA/{filename}")
            for filename in os.listdir(os.path.join(folderpath, "PNEUMONIA"))
        ]
        labels = [1] * len(images_paths)
        normal_paths = [
            os.path.join(folderpath, f"NORMAL/{filename}")
            for filename in os.listdir(os.path.join(folderpath, "NORMAL"))
        ]
        labels.extend([0] * len(normal_paths))
        images_paths.extend(normal_paths)
        self.images_paths = images_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Load image
#        img = np.array(Image.open(self.images_paths[idx]))
        img = cv2.imread(self.images_paths[idx],flags=cv2.IMREAD_GRAYSCALE)
        #img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        augmented = self.transform(image=img)
        return augmented["image"], self.labels[idx]

############################## New transforms using albumentation ######################
def getAlbumentationTransforms(training=False,stretch=False):
    if stretch:
        resize_transform = A.Resize(input_size, input_size)
    else:
        resize_transform = A.Compose([A.LongestMaxSize(input_size), A.PadIfNeeded(input_size, input_size)])
    if(training) :
        a_transforms = A.Compose(
        [
            A.CoarseDropout(p=0.5),
            A.GaussianBlur(p=0.5),
            #A.RandomCrop(),
            resize_transform,
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10,p=0.5),
            A.Normalize(mean=mean,std=std),
            A.ToTensorV2(),

        ]
        )
    else:
        a_transforms= A.Compose(
    [
        resize_transform,
        A.Normalize(mean=mean,std=std),
        A.ToTensorV2(),

    ]
)
    return a_transforms






