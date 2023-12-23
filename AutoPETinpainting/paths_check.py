import os
import time


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
    train_root_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/train'
    ex_names = ['.DS_Store', '@eaDir', '.DS_Store@SynoResource', 'losses', 'models']
    all_train_dirs = get_all_paths(train_root_dir, exclude_names=[])
    image_names = []
    label_names = []
    mask_names = []
    start_time = time.time()
    for all_path in all_train_dirs:
        image_path = os.path.join(all_path, "images")
        label_path = os.path.join(all_path, "labels")
        mask_path = os.path.join(all_path, "masks")
        for image_name in sorted(os.listdir(image_path)):
            image_names.append(os.path.join(image_path, image_name))
            label_names.append(os.path.join(label_path, image_name))
            mask_names.append(os.path.join(mask_path, image_name))
    end_time = time.time()
    print("finished construction! duration time:", end_time - start_time)
    print("length of images", len(image_names))
