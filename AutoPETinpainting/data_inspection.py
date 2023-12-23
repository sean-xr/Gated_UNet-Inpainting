import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
label_dir = "/Users/xiaorui/Documents/test_folder/PETCT_0b98dbe00d/08-11-2002-NA-PET-CT Ganzkoerper  primaer mit KM-83616/labels/slice34.npy"
image_dir = "/Users/xiaorui/Documents/test_folder/PETCT_0b98dbe00d/08-11-2002-NA-PET-CT Ganzkoerper  primaer mit KM-83616/images/slice34.npy"

image_array = np.load(image_dir)
label_array = np.load(label_dir)

plt.subplot(1, 2, 1)
plt.imshow(image_array)  # Use cmap='gray' for grayscale images
plt.title('masked image')

plt.subplot(1, 2, 2)
plt.imshow(label_array)  # Use cmap='gray' for grayscale images
plt.title('original image')

plt.show()
