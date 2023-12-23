"""
run the totalsegmentator on all the test CTres.nii.gz
the resulting masks are stored under ./path/to/each/test/file/segmentations
"""
import os
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator


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


if __name__ == "__main__":
    test_root_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/test'
    ex_names = ['.DS_Store', '@eaDir', '.DS_Store@SynoResource']
    all_test_dirs = get_all_paths(test_root_dir, exclude_names=ex_names)
    for test_dir in all_test_dirs:
        segmentations_dir = os.path.join(test_dir, "segmentations")
        if not os.path.exists(segmentations_dir):
            os.makedirs(segmentations_dir, exist_ok=True)
        ctres_path = os.path.join(test_dir, "CTres.nii.gz")
        print("segmenting into...", segmentations_dir)
        totalsegmentator(ctres_path, segmentations_dir)
        print("finsihed!")
        print()
