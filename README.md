# Gated_UNet-Inpainting
An unofficial implementation for the paper: https://arxiv.org/abs/2305.12358. The Gated-Unet model is trained on slices from the AutoPET dataset.

1. Datset Preparation:
The dataset used for the inpaintainting task is downloaded from:https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287
The downloaded dataset should be DICOM format. To convert the data into nifti format, please refer to the official data conversion script: https://github.com/lab-midas/TCIA_processing. After running the script you should be able to get the NIFTI format data. The data should be automatically organized into the following structure:
![image](https://github.com/sean-xr/Gated_UNet-Inpainting/assets/91930856/3457f51e-e435-48a3-862e-29cd4da9dd5e)

2. Slices Generation:
Since the Gated-UNet should be trained on 2D healthy slices from the NIFTI files, we should run the slice_generation.py to generate healthy slices and the corresponding masks. The workflow for slice_generation.py is as follows: First, it used the 'get_all_paths()' function to get all the paths to all the CTs and segmentation masks in the dataset. Then the script will read all the CTs and their corresponding masks into numpy arrays, meanwhile normalizing the CTs using clamping and min-max normalization. In each healthy slice (where the corresponding mask slice is all zeros), the 'generate_random_square_mask()' function is called to generate random square mask slice. The generaed square mask is multipled by the original healthy slice to become corrupted slice. The corrupted slice will be saved into the '/imagse' sub-directory, while the original slice will be saved into '/labels and the suqare mask will be saved into '/masks'


