import os
import pydicom
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import rotate

# Define paths
image_path = "./HCC_002/09-26-1997-NA-ABDPELVIS-84861/4.000000-Recon 2 LIVER 3 PHASE AP-85789"
seg_path = "./HCC_002/09-26-1997-NA-ABDPELVIS-84861/300.000000-Segmentation-74386/1-1.dcm"

# Load and process image data
img_dcmset = []
acc_num = []
for root, _, filenames in os.walk(image_path):
    for filename in filenames:
        dcm_path = Path(root) / filename
        dicom = pydicom.dcmread(dcm_path, force=True)
        img_dcmset.append(dicom)
        acc_num.append(dicom.AcquisitionNumber)

slice_thickness = dicom.SliceThickness
pixel_spacing = dicom.PixelSpacing[0]


# Only one acquisition
acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]

img_dcmset.sort(key=lambda x: x.ImagePositionPatient[2])
img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)
print('img_pixelarray shape ', img_pixelarray.shape)


mask_dcm = pydicom.dcmread(seg_path)
mask_pixelarray = mask_dcm.pixel_array
print('Mask shape ', mask_pixelarray.shape)

#  Visualize few slides
plt.figure(figsize=(15, 10))

# Iterate to plot images and their corresponding masks for liver and mass
for i in range(1, 4):
    # Calculate index for accessing slices based on the length of the image array
    slice_idx = i * 26

    # Plotting the CT image
    ax = plt.subplot(3, 3, i)  # Changed grid to 3x3 and updated positions
    ax.imshow(img_pixelarray[slice_idx], cmap='gray')
    ax.set_title(f'Image Slice {slice_idx}')
    ax.axis('off')

    # Plotting the liver mask
    ax = plt.subplot(3, 3, i + 3)  # Correct position for liver mask
    ax.imshow(mask_pixelarray[slice_idx,:,:], cmap='gray')  # Assuming first segment is liver, adjust if needed
    ax.set_title(f'Mask Liver Slice {slice_idx}')
    ax.axis('off')

    # Plotting the mass mask
    ax = plt.subplot(3, 3, i + 6)  # Correct position for mass mask
    if len(mask_pixelarray) > 109:  # Check if the mass segment index is valid
        ax.imshow(mask_pixelarray[109 + slice_idx,:,:], cmap='jet')  # Adjusted for mass visualization
    ax.set_title(f'Mask Mass Slice {slice_idx}')
    ax.axis('off')

# Adjust layout to prevent overlap and ensure clear visibility of titles and images
plt.tight_layout()
plt.show()


idx = 78
mass_mask_idx = 109 + idx  # Adjust this based on how your segmentation masks are indexed

if mass_mask_idx < len(mask_pixelarray):
    mass_mask = mask_pixelarray[mass_mask_idx, :, :]
else:
    mass_mask = np.zeros_like(img_pixelarray[idx, :, :])

# Normalize the CT image
ct_slice_normalized = img_pixelarray[idx, :, :] #/ np.max(img_pixelarray[idx, :, :])

# Apply the 'bone' colormap to the CT image
ct_colored = plt.cm.bone(ct_slice_normalized)

# Apply the 'prism' colormap to the mask (make sure mask values are scaled appropriately if not binary)
mask_colored = plt.cm.prism(mass_mask / mass_mask.max())

# Create an alpha mask based on the mass mask (ensure mass_mask is binary or normalized)
alpha = 0.3  # Set the transparency level for the mask areas
alpha_mask = mass_mask.astype(bool).astype(float) * alpha

# Overlay the mask using the alpha channel
overlay_image = ct_colored[..., :3] * (1 - alpha_mask[..., np.newaxis]) + mask_colored[..., :3] * alpha_mask[..., np.newaxis]

# Plotting the resulting overlay image
plt.figure(figsize=(8, 8))
plt.imshow(overlay_image)
plt.title(f'CT with Mass Overlay - Slice {idx}')
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()


def load_scan(path):
    slices = [pydicom.dcmread(path / filename) for filename in os.listdir(path) if filename.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return np.stack([s.pixel_array for s in slices])

def maximum_intensity_projection(volume, axis=1):
    """Computes maximum intensity projection on a specified plane."""
    return np.max(volume, axis=axis)

from scipy.ndimage import rotate


def rotation(volume, angle, axis='z'):
    """Rotates the volume around a specified axis by a given angle."""
    if axis == 'z':
        axes_tuple = (0, 1)  # Rotate in the x-y plane
    elif axis == 'x':
        axes_tuple = (1, 2)  # Rotate in the y-z plane
    elif axis == 'y':
        axes_tuple = (0, 2)  # Rotate in the x-z plane
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return rotate(volume, angle, axes=axes_tuple, reshape=False, mode='nearest')

n_frames = 30
alpha = 0.3  # Transparency for blending mask

frames = []
for angle in np.linspace(0, 360, n_frames, endpoint=False):
    rotated_img = rotation(img_pixelarray, angle, axis='x')
    rotated_mask = rotation(mask_pixelarray, angle, axis='x')

    mip_img = maximum_intensity_projection(rotated_img, axis=1)
    mip_mask = maximum_intensity_projection(rotated_mask, axis=1)

    # Normalize images for blending
    mip_img_norm = (mip_img - mip_img.min()) / (mip_img.max() - mip_img.min())
    mip_maskism_norm = (mip_mask - mip_mask.min()) / (mip_mask.max() - mip_mask.min())

    mip_maskism_norm = mip_maskism_norm[109:2*109]
    cmap_bone = plt.get_cmap('bone')
    cmap_prism = plt.get_cmap('prism')
    mip_img_colored = cmap_bone(mip_img_norm)
    mip_seg_colored = cmap_prism(mip_maskism_norm)

    print(mip_img_colored.shape)
    print(mip_seg_colored.shape)

    # Create blended image
    combined_image = mip_img_colored * (1 - alpha) + mip_seg_colored * alpha
    plt.imshow(combined_image)
    plt.show()
    # combined_image = (combined_image * 255).astype(np.uint8)
    frames.append((combined_image * 255).astype(np.uint8))  # Convert to 8-bit for GIF
# Save as GIF using imageio, setting correct aspect ratio
fig, ax = plt.subplots()
for idx, frame in enumerate(frames):
    ax.clear()
    ax.imshow(frame, cmap='gray', aspect=slice_thickness / pixel_spacing)

    plt.axis('off')
    plt.savefig(f'frame_{idx:03d}.png', bbox_inches='tight', pad_inches=0)

# Convert saved images to GIF
imageio.mimsave('rotating_mip_animation.gif', [imageio.v2.imread(f'frame_{i:03d}.png') for i in range(n_frames)], fps=10)

print("Animation saved as 'rotating_mip_animation.gif'")

