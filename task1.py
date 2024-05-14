import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from scipy.ndimage import rotate

# Define paths
IMAGE_PATH = "./HCC_002/09-26-1997-NA-ABDPELVIS-84861/4.000000-Recon 2 LIVER 3 PHASE AP-85789"
SEG_PATH = "./HCC_002/09-26-1997-NA-ABDPELVIS-84861/300.000000-Segmentation-74386/1-1.dcm"

def load_scan(path):
    path = Path(path)
    slices = sorted(os.listdir(path), key=lambda x: float(pydicom.dcmread(path / x).ImagePositionPatient[2]))
    return np.stack([pydicom.dcmread(path / f).pixel_array for f in slices])

def rotation(volume, angle, axis='z'):
    axes_tuple = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}.get(axis, (0, 1))
    return rotate(volume, angle, axes=axes_size, reshape=False, mode='nearest')

def maximum_intensity_projection(volume, axis=1):
    return np.max(volume, axis=axis)

# Load data
img_pixelarray = load_scan(IMAGE_PATH)
mask_pixelarray = load_matrix(SEG_PATH)

slice_thickness = pydicom.dcmread(IMAGE_PATH).SliceThickness
pixel_spacing = pydicom.dcmread(IMAGE_PATH).PixelSpacing[0]

# Visualize initial slices
plt.figure(figsize=(15, 10))
for i in range(3):
    idx = i * 26  # Adjust based on the dataset length
    plt.subplot(3, 3, i+1)
    plt.imshow(img_pixelarray[idx], cmap='bone')
    plt.title(f'CT Slice {idx}')
    plt.axis('off')
    plt.subplot(3, 3, i+4)
    plt.imshow(mask_pixelarray[idx], cmap='prism')
    plt.title(f'Mask Slice {idx}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Create animations
n_frames = 30
alpha = 0.3
frames = []
for angle in np.linspace(0, 360, n_frames, endpoint=False):
    rotated_img = rotation(img_pixelarray, angle, 'x')
    rotated_mask = rotation(mask_pixelarray, angle, 'x')
    mip_img = maximum_intensity_projection(rotated_img, axis=1)
    mip_mask = maximum_intensity_projection(rotated_mask, axis=1)

    # Combine images
    mip_img_norm = (mip_img - mip_img.min()) / (mip_img.max() - mip_img.min())
    mip_mask_norm = (mip_mask - mip_mask.min()) / (mip_mask.max() - mip_mask.min())

    # Apply colormaps
    img_colored = plt.cm.bone(mip_img_norm)
    mask_colored = plt.cm.prism(mip_mask_norm)

    # Blend images
    combined_image = img_colored * (1 - alpha) + mask_colored * alpha
    frames.append((combined_image * 255).astype(np.uint8))

# Save frames to GIF
imageio.mimsave('rotating_mip_animation.gif', frames, fps=10)

print("Animation saved as 'rotating_mip_animation.gif'")
