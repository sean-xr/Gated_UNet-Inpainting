from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch
from dataset2 import InPaintingDataset2
from dataset2 import RandContrast, RandVerFlip, RandHorFlip, RandRotation, Resize, ToTensor, Normalize, Crop
import time


batch_size = 16

train_transform = Compose([ToTensor(),
                           Crop()])

valid_transform = Compose([ToTensor(),
                           Crop()])

train_root_dir = "/Users/xiaorui/Documents/test_folder"

train_set = InPaintingDataset2(train_root_dir, transform=train_transform)

train_loader = DataLoader(train_set, batch_size, shuffle=False)

device = torch.device("cuda")

if __name__ == "__main__":
    for i, data in enumerate(train_loader):
        print("batch", i)
        img, seg, msk = data





