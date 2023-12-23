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





def check_shape(input_dir):
    print()
    print("checking shapes of dir:", input_dir)
    ct_res_array = nib.load(os.path.join(input_dir, "CTres.nii.gz")).get_fdata()
    pet_array = nib.load(os.path.join(input_dir, "PET.nii.gz")).get_fdata()
    seg_array = nib.load(os.path.join(input_dir, "SEG.nii.gz")).get_fdata()
    print("ct_res_shape:", ct_res_array.shape)
    print("pet shape:", pet_array.shape)
    print("seg shape:", seg_array.shape)


def check_min_max(input_dir, output_dir):
    """
    :param input_dir: string
    :param output_dir: string
    :return: None
    """
    # Load the CTres_array from the input directory
    ct_res_array = nib.load(os.path.join(input_dir, "CTres.nii.gz")).get_fdata()

    # Compute the max and min values
    max_value = np.max(ct_res_array)
    min_value = np.min(ct_res_array)

    # Write max and min values to a text file in the output directory
    output_filepath = os.path.join(output_dir, "min_max_values.txt")
    mode = 'a' if os.path.exists(output_filepath) else 'w'
    with open(output_filepath, mode) as file:
        file.write(f'Max Value: {max_value}\n')
        file.write(f'Min Value: {min_value}\n')


if __name__ == "__main__":
    output_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/datacheck'
    train_root_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/train'
    #test_root_dir = '/home/polyaxon-data/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions/test'
    ex_names = ['.DS_Store']
    all_train_dirs = get_all_paths(train_root_dir, exclude_names=ex_names)
    #all_test_dirs = get_all_paths(test_root_dir, exclude_names=ex_names)
    for train_dir in all_train_dirs:
        check_shape(train_dir)
        check_min_max(train_dir, output_dir)

