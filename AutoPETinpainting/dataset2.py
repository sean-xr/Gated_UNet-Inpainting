import os
import numpy as np

from skimage import io
from torchvision.transforms import RandomRotation, Compose, ToTensor, PILToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import adjust_contrast, vflip, hflip, rotate, to_tensor, resize, normalize, crop


def get_all_paths(root_dir, exclude_names=[]):
    all_paths = []
    sub_dirs = os.listdir(root_dir)
    for sub_dir in sub_dirs:
        if sub_dir not in exclude_names:
            sub_sub_dirs = os.listdir(os.path.join(root_dir, sub_dir))
            for sub_sub_dir in sub_sub_dirs:
                if sub_sub_dir not in exclude_names:
                    all_paths.append(os.path.join(root_dir, sub_dir, sub_sub_dir))
    return all_paths

class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        img, seg, mask = sample["image"], sample["label"], sample["mask"]
        img = to_tensor(img)
        seg = to_tensor(seg)
        msk = to_tensor(mask)
        return {"image": img, "label": seg, "mask": msk}

class Resize:
    def __init__(self, size, inter):
        self.size = size
        self.inter = inter

    def __call__(self, sample):
        img, seg = sample["image"], sample["mask"]
        img = resize(img, self.size, self.inter)
        seg = resize(seg, self.size, self.inter)
        return {"image": img, "mask": seg}


class Crop:
    def __init__(self):
        pass

    def __call__(self, sample):
        img, seg, mask = sample["image"], sample["label"], sample["mask"]
        img = crop(img, 8, 8, 384, 384)
        seg = crop(seg, 8, 8, 384, 384)
        msk = crop(mask, 8, 8, 384, 384)
        return {"image": img, "label": seg, "mask": msk}


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, seg = sample["image"], sample["mask"]
        img = normalize(img, self.mean, self.std)
        return {"image": img, "mask": seg}


class RandContrast:
    def __init__(self, lower_bound=0.8, upper_bound=1.2):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, sample):
        img, seg = sample["image"], sample["mask"]
        contrast_factor = np.random.uniform(self.lower_bound, self.upper_bound)
        return {"image": adjust_contrast(img, contrast_factor), "mask": seg}


class RandHorFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        img, seg = sample["image"], sample["mask"]
        if np.random.rand() <= self.prob:
            img = hflip(img)
            seg = hflip(seg)
        return {"image": img, "mask": seg}


class RandVerFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        img, seg = sample["image"], sample["mask"]
        if np.random.rand() <= self.prob:
            img = vflip(img)
            seg = vflip(seg)
        return {"image": img, "mask": seg}


class RandRotation90:
    def __init__(self, inter, expand=False, center=None, fill=None):
        self.inter = inter
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, sample):
        angle = np.random.choice([90, 180, 270])
        rot_ori = np.random.choice([1, -1])
        angle = int(angle) if rot_ori == 1 else int(360 - angle)

        img, seg = sample["image"], sample["mask"]
        img = rotate(img, angle, self.inter, self.expand, self.center, self.fill)
        seg = rotate(seg, angle, self.inter, self.expand, self.center, self.fill)
        return {"image": img, "mask": seg}


class RandRotation:
    def __init__(self, lower_degree, upper_degree, inter, expand=False, center=None, fill=None):
        self.lower_degree = lower_degree
        self.upper_degree = upper_degree
        self.inter = inter
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, sample):
        img, seg = sample["image"], sample["mask"]
        angle = RandomRotation.get_params([self.lower_degree, self.upper_degree])
        img = rotate(img, angle, self.inter, self.expand, self.center, self.fill)
        seg = rotate(seg, angle, self.inter, self.expand, self.center, self.fill)
        return {"image": img, "mask": seg}


class NeuronDataset(Dataset):

    def __init__(self, root_dir, num_slices, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_slices = num_slices

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        volume = io.imread(os.path.join(self.root_dir, "images", "train-volume.tif"))
        label = io.imread(os.path.join(self.root_dir, "labels", "train-labels.tif"))

        data_slice = volume[idx, :, :]
        label_slice = label[idx, :, :]

        if self.transform is not None:
            sample = {"image": data_slice, "mask": label_slice}
            sample = self.transform(sample)
            data_slice, label_slice = sample["image"], sample["mask"]

        label_slice = 1 - label_slice
        return data_slice, label_slice


class InPaintingDataset2(Dataset):

    def __init__(self, root_dir, transform=None, down_sample=4):
        self.root_dir = root_dir
        self.transform = transform
        self.down_sample = down_sample
        self.all_paths = get_all_paths(root_dir,
                                       exclude_names=['.DS_Store', '@eaDir', '.DS_Store@SynoResource', 'losses',
                                                      'models'])
        self.exclude_names = ['.DS_Store', '@eaDir', '.DS_Store@SynoResource', 'losses', 'models']
        self.image_names = []
        self.label_names = []
        self.mask_names = []
        for all_path in self.all_paths:
            image_path = os.path.join(all_path, "images")
            label_path = os.path.join(all_path, "labels")
            mask_path = os.path.join(all_path, "masks")
            images_names = sorted(os.listdir(image_path))
            for i in range(0, len(images_names), self.down_sample):
                if images_names[i] not in self.exclude_names:
                    self.image_names.append(os.path.join(image_path, images_names[i]))
                    self.label_names.append(os.path.join(label_path, images_names[i]))
                    self.mask_names.append(os.path.join(mask_path, images_names[i]))
        print("length of images", len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = np.load(self.image_names[idx])
        seg = np.load(self.label_names[idx])
        msk = np.load(self.mask_names[idx])
        print("loading", self.image_names[idx])
        if self.transform is not None:
            sample = {"image": img, "label": seg, "mask": msk}
            sample = self.transform(sample)
            img, seg, msk = sample["image"], sample["label"], sample["mask"]

        return img, seg, msk
