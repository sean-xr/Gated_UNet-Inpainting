import os
import nibabel as nib
import numpy as np


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


def ct_preprocessing(ct_data, min_ct_value=-1000., max_ct_value=500.):
    ct_data_clamped = np.clip(ct_data, min_ct_value, max_ct_value)
    ct_data_normalized = (ct_data_clamped - min_ct_value) / (max_ct_value - min_ct_value)
    return ct_data_normalized


def generate_random_square_mask(image_size, min_size, max_size, boundary_value, min_num_squares, max_num_squares):
    """
    Generate a random binary mask with a random number of squares for a 2-channel image.

    Args:
        image_size (tuple): Size of the image.
        min_size (int): Minimum size of the square mask.
        max_size (int): Maximum size of the square mask.
        boundary_value (int): Value that limits the center of the squares to be close to the center.
        min_num_squares (int): Minimum number of squares to generate.
        max_num_squares (int): Maximum number of squares to generate.

    Returns:
        np.ndarray: Binary mask of shape (image_size, image_size, 2) with a random number of squares for each channel.
    """
    mask = np.full((image_size[0], image_size[1]), fill_value=1, dtype=np.uint8)

    num_squares = np.random.randint(min_num_squares, max_num_squares + 1)  # Random number of squares

    for _ in range(num_squares):
        # Randomly determine the size of the square mask
        mask_size = np.random.randint(min_size, max_size + 1)

        # Randomly determine the position of the square mask (centered)
        x_center = np.random.randint(mask_size // 2 + boundary_value, image_size[0] - mask_size // 2 - boundary_value)
        y_center = np.random.randint(mask_size // 2 + boundary_value, image_size[1] - mask_size // 2 - boundary_value)

        # Fill the square region with zeros (black) in both channels
        x_start = x_center - mask_size // 2
        x_end = x_start + mask_size
        y_start = y_center - mask_size // 2
        y_end = y_start + mask_size

        mask[x_start:x_end, y_start:y_end] = 0

    return mask


def slice_mask_generation(seg_array, ct_norm, root_dir):
    print(f"generating in directory:{root_dir}")
    # Iterate through axial slices, select where SEG is 0, and concatenate PET and CT
    for i in range(seg_array.shape[2]):
        if np.all(seg_array[:, :, i] == 0.):  # Check if all values in the SEG slice are 0
            ct_slice = ct_norm[:, :, i]
            mask = generate_random_square_mask(ct_slice.shape, min_size=20, max_size=32,
                                               boundary_value=110, min_num_squares=3, max_num_squares=5)
            masked_image = ct_slice * mask
            # Save the concatenated array to the output subfolder
            labels_save_path = os.path.join(root_dir, "labels", f"slice{i}.npy")
            masks_save_path = os.path.join(root_dir, "masks", f"slice{i}.npy")
            images_save_path = os.path.join(root_dir, "images", f"slice{i}.npy")
            np.save(labels_save_path, ct_slice)
            np.save(masks_save_path, mask)
            np.save(images_save_path, masked_image)
        else:
            print(f"dispose slice{i}")


if __name__ == "__main__":
    #train_root_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/train'
    train_root_dir = "/Users/xiaorui/Documents/test_folder"
    ex_names = ['.DS_Store', '@eaDir', '.DS_Store@SynoResource']
    all_train_dirs = get_all_paths(train_root_dir, exclude_names=ex_names)
    for train_dir in all_train_dirs:

        labels_dir = os.path.join(train_dir, "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir, exist_ok=True)
        masks_dir = os.path.join(train_dir, "masks")
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir, exist_ok=True)
        images_dir = os.path.join(train_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)

        ctres_path = os.path.join(train_dir, "CTres.nii.gz")
        seg_path = os.path.join(train_dir, "SEG.nii.gz")

        ctres_array = nib.load(ctres_path).get_fdata()
        seg_array = nib.load(seg_path).get_fdata()
        ct_norm = ct_preprocessing(ctres_array, min_ct_value=-1000., max_ct_value=500.)

        slice_mask_generation(seg_array, ct_norm, train_dir)

