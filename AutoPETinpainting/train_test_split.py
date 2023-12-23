import os
import shutil
import random

# Define the path to the main directory
main_directory = '/Volumes/polyaxon/data1/rui_xiao/Original_Dataset/NIFTI/FDG-PET-CT-Lesions'

# Check if train and test folders exist, create them if not
train_folder = os.path.join(main_directory, 'train')
test_folder = os.path.join(main_directory, 'test')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# List all folders in the main directory
all_folders = [folder for folder in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, folder))]

# Randomly sample 100 folders for the test set
test_folders = random.sample(all_folders, 100)

# Move folders to train or test based on the random sampling
for folder in all_folders:
    source_path = os.path.join(main_directory, folder)
    if folder in test_folders:
        destination_path = os.path.join(test_folder, folder)
    else:
        destination_path = os.path.join(train_folder, folder)

    # Move the folder
    shutil.move(source_path, destination_path)

print("Folders have been randomly sampled and moved to train/test.")
