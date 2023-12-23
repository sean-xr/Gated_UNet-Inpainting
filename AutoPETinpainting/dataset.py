
import os
import glob
import copy
import PIL.Image
import numpy as np
import torchvision.transforms
from torchvision.transforms import ToTensor

from skimage import io
from typing import Tuple, List, Union

import time
import torch
import torch.nn as nn

from torch.utils.data import Dataset


from torchvision.transforms.functional import to_tensor
from torch.nn.functional import softmax

Tensor = torch.Tensor
Module = nn.Module
Array = np.ndarray


class VesselDataset3D(Dataset):
    def __init__(self,
                 root_dir: str,
                 image_names: List,
                 label_names: List,
                 volume_size: Tuple[int, int, int],
                 device,
                 num_new_samples,
                 fold_number: int,
                 mode: str = "half",
                 exist_labels: bool = False,
                 transform=None):
        super(VesselDataset3D, self).__init__()

        """

        Parameters:
        -----------
        - root_dir: the directory of volumes and corresponding labels
        - volume_size: a tuple of 3 components to specify size of input volume (H, W, D)
        """

        self.root_dir = root_dir
        self.cont_save_dir = os.path.join(self.root_dir, "cont")
        
        self.epoch = 0
        self.device = device
        self.exist_labels = exist_labels
        self.volume_size = volume_size
        self.num_new_samples = num_new_samples

        self.mode = mode
        self.transform = transform
        self.fold_number = fold_number

        self.image_names = image_names  # we move the image_names&label_names to the training files for k-fold cross validation
        self.label_names = label_names
        self.data_dict = [{"image": image_name, "label": label_name} for image_name, label_name in
                          zip(image_names, label_names)]

        self.heights, self.widths, self.depths, self.directions = self.create_indices_and_directions()

    def create_indices_and_directions(self) -> Tuple[Array, Array, Array, Array]:

        h, w, d = 1, 1, 1

        heights = 3 * ([h - 1] * 3 + [h] * 3 + [h + 1] * 3)
        widths = 3 * ([w - 1, w, w + 1] * 3)
        depths = [d - 1] * 9 + [d] * 9 + [d + 1] * 9

        heights = np.array(heights[:13] + heights[14:])
        widths = np.array(widths[:13] + widths[14:])
        depths = np.array(depths[:13] + depths[14:])

        if self.mode == "full":
            directions = np.array([i for i in range(0, 26)])
        else:
            directions = np.array([i for i in range(0, 13)])
            directions = np.concatenate((directions, directions[::-1]))

        return heights, widths, depths, directions

    def fill_cont_labels(self, seg_label: Tensor):

        """

        :param seg_label: it is 3D Tensor of HxWxD shape
        :return:
        """

        shape = seg_label.shape
        channels = 26 if self.mode == "full" else 13
        cont_label = np.zeros((channels, *shape), dtype=np.uint8)

        foreground_coordinates = torch.argwhere(seg_label == 1).to(torch.int32)
        print("New Label")
        print(len(foreground_coordinates))
        start1 = time.time()
        for index, coor in enumerate(foreground_coordinates):
            i, j, k = coor
            heights = (self.heights - 1) + i
            widths = (self.widths - 1) + j
            depths = (self.depths - 1) + k
            indices = self.find_neighbors(heights, widths, depths, seg_label)
            directions = self.directions[indices]
            for d in directions:
                cont_label[d, i, j, k] = 255
            print(index)
        end1 = time.time()
        print("duration of new label", end1 - start1)
        return cont_label

    def count_epoch(self):
        self.epoch += 1

    def find_neighbors(self, heights: Array, widths: Array, depths: Array, label: Tensor) -> List:
        H, W, D = self.volume_size
        neighbor_indices = []

        indices = np.array([i for i in range(0, 26)])
        for h, w, d, i in zip(heights, widths, depths, indices):
            in_bound = ((0 <= h < H) and (0 <= w < W) and (0 <= d <= D))

            if in_bound:
                foreground = (label[h, w, d] == 1)
            else:
                foregrund = False

            if in_bound and foreground:
                neighbor_indices.append(i)
        return neighbor_indices

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        sample_dict = self.data_dict[idx]

        if self.transform is not None:
            sample = self.transform(sample_dict)
            vol, seg = sample["image"], sample["label"]
            # vol = torch.squeeze(vol, dim=0)
            seg = torch.squeeze(seg, dim=0)

            if self.exist_labels:
                cont = torch.load(os.path.join(self.cont_save_dir, f"{idx}.pt"))
                cont = ToTensor()(cont)
            else:
                if self.epoch == 0:
                    cont = self.fill_cont_labels(seg)  # FIXME: map to tensor by to_tensor function
                    torch.save(cont, os.path.join(self.cont_save_dir, f"{idx}.pt"))
                    cont = ToTensor()(cont)
                else:
                    cont = torch.load(os.path.join(self.cont_save_dir, f"{idx}.pt"))
                    cont = ToTensor()(cont)

        # CHECK: Their type has to be float for graph generation
        #vol = vol.to(self.device)
        #seg = seg.to(self.device)
        #cont = cont.to(self.device)

        return vol, seg, cont
