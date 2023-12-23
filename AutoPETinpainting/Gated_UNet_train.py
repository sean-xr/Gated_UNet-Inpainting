import numpy as np
import torch
import monai
import torchmetrics as tm
import torchsummary
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import os
from unet import GatedUNet

from dataset2 import InPaintingDataset2
from dataset2 import RandContrast, RandVerFlip, RandHorFlip, RandRotation, Resize, ToTensor, Normalize, Crop
from loss import valid_loss_fn, hole_loss_fn, tv_loss_fn, PerceptualLoss, StyleLoss, LPLoss

batch_size = 32

train_transform = Compose([ToTensor(),
                           Crop()])

valid_transform = Compose([ToTensor(),
                           Crop()])

train_root_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/train'
model_save_dir = os.path.join(train_root_dir, "models")
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)
loss_save_dir = os.path.join(train_root_dir, "losses")
if not os.path.exists(loss_save_dir):
    os.makedirs(loss_save_dir, exist_ok=True)

train_set = InPaintingDataset2(train_root_dir, transform=train_transform)
train_loader = DataLoader(train_set, batch_size, shuffle=False)

save_model = True
model_id = abs(round(np.random.normal(0, 1), 5))

lr = 0.0001
num_epochs = 10
best_valid_loss = float("inf")

model_name = "GatedUnet"
dataset_name = "AutoPET"
loss_log = "combine Loss"
print(model_id)

device = torch.device("cuda")
print(device)
model = GatedUNet(in_channels=1, out_channels=1, bilinear=False)
model.to(device)
model.train()

mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
perceptual_loss_fn = PerceptualLoss(device=device)
style_loss_fn = StyleLoss(device=device)
lp_loss_fn = LPLoss(max_level=3)
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
acc_fn = tm.classification.BinaryAccuracy().to(device)
pre_fn = tm.classification.BinaryPrecision().to(device)
rec_fn = tm.classification.BinaryRecall().to(device)

if __name__ == "__main__":
    all_train_losses = []
    all_valid_losses = []
    for epoch in range(num_epochs):

        model.train()
        train_losses, valid_losses = [], []
        for i, data in enumerate(train_loader):
            imgs, labels, masks = data
            imgs = imgs.to(device).to(torch.float32)  # images shape Nx2x384x384
            labels = labels.to(device).to(torch.float32)  # labels shape Nx2x384x384
            masks = masks.to(device)  # masks shape Nx2x384x384

            optim.zero_grad()
            preds = model(imgs)  # predictions in the shape of Nx1x512x512
            valid_loss = valid_loss_fn(preds, labels, masks)
            hole_loss = hole_loss_fn(preds, labels, masks)
            lp_loss = lp_loss_fn(preds, labels)
            loss = 30 * valid_loss + 240 * hole_loss + 20 * lp_loss
            loss.backward()
            optim.step()
            train_losses.append(loss.detach().item())

            model.eval()
            with torch.no_grad():
                val_preds = model(imgs)
                val_valid_loss = valid_loss_fn(val_preds.detach(), labels.detach(), masks.detach())
                val_hole_loss = hole_loss_fn(val_preds.detach(), labels.detach(), masks.detach())
                val_lp_loss = lp_loss_fn(val_preds.detach(), labels.detach())
                val_loss = 30 * val_valid_loss + 240 * val_hole_loss + 20 * val_lp_loss
                valid_losses.append(val_loss.detach().item())

        ave_train_loss = sum(train_losses) / len(train_losses)
        ave_valid_loss = sum(valid_losses) / len(valid_losses)

        if save_model and epoch > 80 and epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_{model_id}_{epoch}.pt"))

        print(f"Epoch {epoch} train loss: ", ave_train_loss)
        print(f"Epoch {epoch} valid loss: ", ave_valid_loss)
        print()

        all_train_losses.append(ave_train_loss)
        all_valid_losses.append(ave_valid_loss)

        acc_fn.reset()
        pre_fn.reset()
        rec_fn.reset()
    np.save(os.path.join(loss_save_dir, f"train_{model_id}.npy"), np.array(all_train_losses))
    np.save(os.path.join(loss_save_dir, f"valid_{model_id}.npy"), np.array(all_valid_losses))
