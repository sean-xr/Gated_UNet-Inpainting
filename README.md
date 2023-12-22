# Gated_UNet-Inpainting
An unofficial implementation for the paper: https://arxiv.org/abs/2305.12358. The Gated-Unet model is trained on slices from the AutoPET dataset.

1. Datset Preparation:
The dataset used for the inpaintainting task is downloaded from:https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287
The downloaded dataset should be DICOM format. To convert the data into nifti format, please refer to the official data conversion script: https://github.com/lab-midas/TCIA_processing. After running the script you should be able to get the NIFTI format data. The data should be automatically organized into the following structure:
![image](https://github.com/sean-xr/Gated_UNet-Inpainting/assets/91930856/3457f51e-e435-48a3-862e-29cd4da9dd5e)

