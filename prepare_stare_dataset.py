import os
import h5py
import numpy as np
from PIL import Image
from glob import glob
import cv2

# --- Configuration ---
# IMPORTANT: Update this path to where you have unzipped the STARE dataset
STARE_DATA_PATH = './stare_dataset/' 

# Path where the script will save the new HDF5 files
OUTPUT_PATH = './STARE_datasets_training_testing/'

# --- Ground Truth Selection ---
# STARE provides two sets of labels. 'ah' is the primary one.
IMAGE_FOLDER = 'stare-images'
LABEL_FOLDER = 'labels-ah'
LABEL_SUFFIX = '.ah.ppm'

def create_fov_mask(image_path, threshold=15):
    """
    Creates a Field of View (FOV) mask from the original color image.
    The STARE dataset doesn't provide masks, so we generate them by finding
    the circular region of the retina against the black background.
    """
    # Read the image in its original color format
    img = cv2.imread(image_path)
    # Convert to grayscale to make thresholding easier
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to separate the retina from the black background
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Use morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def write_hdf5(data, outfile):
    """
    Writes the provided image, ground truth, and mask data to a single HDF5 file.
    """
    with h5py.File(outfile, "w") as f:
        f.create_dataset("imgs", data=data['imgs'], compression="gzip")
        f.create_dataset("groundTruth", data=data['groundTruth'], compression="gzip")
        f.create_dataset("borderMasks", data=data['borderMasks'], compression="gzip")
    print(f"Successfully created HDF5 file: {outfile}")

def prepare_dataset():
    """
    Main function to process the STARE dataset and save it in HDF5 format.
    """
    # Create the output directory if it doesn't already exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Find all the original image paths
    image_paths = sorted(glob(os.path.join(STARE_DATA_PATH, IMAGE_FOLDER, '*.ppm')))

    # Prepare empty lists to hold the data for all images
    all_images = []
    all_gts = []
    all_masks = []

    for img_path in image_paths:
        print(f"Processing: {os.path.basename(img_path)}")
        
        # --- Load Original Image (as 3-channel RGB) ---
        img = Image.open(img_path).convert('RGB')
        all_images.append(np.array(img))

        # --- Load Ground Truth ---
        base_name = os.path.basename(img_path).split('.')[0]
        gt_path = os.path.join(STARE_DATA_PATH, LABEL_FOLDER, f"{base_name}{LABEL_SUFFIX}")
        gt = Image.open(gt_path).convert('L') # Convert GT to grayscale (it's single channel)
        gt_array = np.array(gt)
        # Binarize the ground truth: vessel pixels become 1, background becomes 0
        gt_binary = np.where(gt_array > 128, 1, 0)
        all_gts.append(gt_binary)

        # --- Create and Binarize FOV Mask ---
        mask = create_fov_mask(img_path)
        mask_binary = np.where(mask > 128, 1, 0)
        all_masks.append(mask_binary)

    # --- Convert lists to correctly shaped NumPy arrays ---
    
    # Correctly stack the list of (H, W, 3) images into a single (N, H, W, 3) array
    stacked_images = np.stack(all_images, axis=0)
    # Now, transpose the axes to be (N, 3, H, W) as expected by PyTorch
    final_images = stacked_images.transpose(0, 3, 1, 2)

    # For ground truths and masks, they are (H, W), so we stack them to (N, H, W)
    # and then add the channel dimension to make them (N, 1, H, W)
    final_gts = np.expand_dims(np.stack(all_gts, axis=0), axis=1)
    final_masks = np.expand_dims(np.stack(all_masks, axis=0), axis=1)

    # Normalize image pixel values to be between 0.0 and 1.0
    final_images = final_images / 255.0
    
    # --- Prepare data dictionary for saving ---
    # For simplicity, we'll put all STARE images into a single "training" set.
    # A proper experiment would require a defined train/test split.
    stare_data = {
        'imgs': final_images,
        'groundTruth': final_gts,
        'borderMasks': final_masks
    }

    # Write the final data to an HDF5 file
    output_filename = os.path.join(OUTPUT_PATH, 'STARE_dataset_all.hdf5')
    write_hdf5(stare_data, output_filename)
    
if __name__ == '__main__':
    prepare_dataset()