import os
import pydicom
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import imageio


def normalize(input_array):
    amin = np.amin(input_array)
    amax = np.amax(input_array)
    return (input_array - amin) / (amax - amin)


def load_image_data(image_path):
    img_dcmset = []
    for root, _, filenames in os.walk(image_path):
        for filename in filenames:
            dcm_path = Path(root) / filename
            dicom = pydicom.dcmread(dcm_path, force=True)
            img_dcmset.append(dicom)
    return img_dcmset


def process_image_data(img_dcmset):
    slice_thickness = img_dcmset[0].SliceThickness
    pixel_spacing = img_dcmset[0].PixelSpacing[0]

    # Making sure that there is only one acquisition
    acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
    img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]
    img_dcmset.sort(key=lambda x: x.ImagePositionPatient[2])
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)

    return img_dcmset,img_pixelarray, slice_thickness, pixel_spacing


def load_segmentation_data(seg_path):
    mask_dcm = pydicom.dcmread(seg_path)
    seg_data = mask_dcm.pixel_array
    return seg_data


def visualize_slices(img_pixelarray, seg_data):
    plt.figure(figsize=(15, 10))
    for i in range(1, 4):
        slice_idx = i * 26

        ax = plt.subplot(3, 3, i)
        ax.imshow(img_pixelarray[slice_idx], cmap='gray')
        ax.set_title(f'Image Slice {slice_idx}')
        ax.axis('off')

        ax = plt.subplot(3, 3, i + 3)
        ax.imshow(seg_data[slice_idx, :, :], cmap='gray')
        ax.set_title(f'Mask Liver Slice {slice_idx}')
        ax.axis('off')

        ax = plt.subplot(3, 3, i + 6)
        if len(seg_data) > 109:
            ax.imshow(seg_data[109 + slice_idx, :, :], cmap='jet')
        ax.set_title(f'Mask Mass Slice {slice_idx}')
        ax.axis('off')
    plt.savefig(f'results/task1/viz_slices.png', bbox_inches='tight', pad_inches=0)
    plt.tight_layout()
    plt.show()


def create_overlay_image(img_pixelarray, seg_data, idx):
    mass_mask_idx = 109 + idx
    if mass_mask_idx < len(seg_data):
        mass_mask = seg_data[mass_mask_idx, :, :]
    else:
        mass_mask = np.zeros_like(img_pixelarray[idx, :, :])

    ct_slice_normalized = normalize(img_pixelarray[idx, :, :])
    mass_mask = (mass_mask > 0).astype(float)

    ct_colored = plt.cm.bone(ct_slice_normalized)
    mask_colored = plt.cm.prism(mass_mask)

    alpha = 0.3
    alpha_mask = mass_mask * alpha

    overlay_image = ct_colored[..., :3] * (1 - alpha_mask[..., np.newaxis]) + mask_colored[..., :3] * alpha_mask[
        ..., np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_image)
    plt.savefig(f'results/task1/overlay.png', bbox_inches='tight', pad_inches=0)
    plt.title(f'CT with Mass Overlay - Slice {idx}')
    plt.axis('off')
    plt.show()


def create_segmentation_masks(seg_dcm, ct_slices, ignore=False):
    seg_data = seg_dcm.pixel_array
    seg_shape = seg_data.shape
    ref_segs = seg_dcm.PerFrameFunctionalGroupsSequence

    valid_masks = {}
    for i, ref_seg in enumerate(ref_segs):
        ref_ct = ref_seg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        seg_number = ref_seg.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        img_position = ref_seg.PlanePositionSequence[0].ImagePositionPatient
        valid_masks.setdefault(ref_ct, {})[seg_number] = {"idx": i, "image_position": img_position}

    seg_3d = np.zeros((109, seg_shape[1], seg_shape[2]), dtype=np.int32)

    for z, ct_slice in enumerate(ct_slices):
        if ct_slice.SOPInstanceUID in valid_masks:
            for seg_number, info in valid_masks[ct_slice.SOPInstanceUID].items():
                seg_slice = seg_data[info["idx"]]
                seg_3d[z][seg_slice == 1] = int(seg_number)

    if ignore:
        seg_3d = np.zeros((109, seg_shape[1], seg_shape[2]), dtype=np.int32)

        seg_3d[seg_data[:109] == 1] = 1  # Liver
        seg_3d[seg_data[109:218] == 1] = 2  # Tumor
        seg_3d[seg_data[218:327] == 1] = 3  # Vein
        seg_3d[seg_data[327:436] == 1] = 4  # Aorta

    return seg_3d


def maximum_intensity_projection(image, axis=0):
    return np.max(image, axis=axis)


def create_gif(img_pixelarray, seg_3d, slice_thickness, pixel_spacing, n_frames=30, alpha=0.3):
    frames = []
    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Liver (red)
        [0.0, 1.0, 0.0, 1.0],  # Tumor (green)
        [0.0, 0.0, 1.0, 1.0],  # Vein (blue)
        [1.0, 1.0, 0.0, 1.0]  # Aorta (yellow)
    ]

    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        rotated_img = rotate(img_pixelarray, angle, axes=(1, 2), reshape=False)
        rotated_seg = rotate(seg_3d, angle, axes=(1, 2), reshape=False)

        mip_img = maximum_intensity_projection(rotated_img, axis=1)
        mip_seg = maximum_intensity_projection(rotated_seg, axis=1)

        mip_img_norm = normalize(mip_img)

        cmap_bone = plt.get_cmap('bone')
        mip_img_colored = cmap_bone(mip_img_norm)

        combined_image = mip_img_colored[..., :3]

        for idx, color in enumerate(colors):
            mask = (mip_seg == (idx + 1)).astype(float)
            mask_colored = color[:3] * mask[..., np.newaxis]
            combined_image = combined_image * (1 - alpha * mask[..., np.newaxis]) + mask_colored * alpha
        plt.imshow(combined_image)
        plt.show()
        combined_image_uint8 = (combined_image * 255).astype(np.uint8)
        frames.append(combined_image_uint8)

        fig, ax = plt.subplots()
        ax.clear()
        ax.imshow(combined_image_uint8, cmap='prism', aspect=slice_thickness / pixel_spacing)
        plt.axis('off')
        plt.savefig(f'frames/frame_{i:03d}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    imageio.mimsave('results/task1/rotating_mip_animation.gif',
                    [imageio.v2.imread(f'frames/frame_{i:03d}.png') for i in range(n_frames)], fps=10)
    print("Animation saved as 'rotating_mip_animation.gif'")


def main():
    image_path = "./HCC_002/09-26-1997-NA-ABDPELVIS-84861/4.000000-Recon 2 LIVER 3 PHASE AP-85789"
    seg_path = "./HCC_002/09-26-1997-NA-ABDPELVIS-84861/300.000000-Segmentation-74386/1-1.dcm"

    img_dcmset = load_image_data(image_path)
    img_dcmset, img_pixelarray, slice_thickness, pixel_spacing = process_image_data(img_dcmset)
    seg_data = load_segmentation_data(seg_path)

    visualize_slices(img_pixelarray, seg_data)
    create_overlay_image(img_pixelarray, seg_data, idx=78)

    seg_3d = create_segmentation_masks(pydicom.dcmread(seg_path), img_dcmset,True)

    create_gif(img_pixelarray, seg_3d, slice_thickness, pixel_spacing)


if __name__ == "__main__":
    main()
