import os
import cv2
from tqdm import tqdm
import argparse

cwd = os.getcwd()

data_dir = cwd + '/data' + '/brown_bag'
images_dir = data_dir + '/raw_image'
processed_img_dir = data_dir + 'images'
print(images_dir)
print(processed_img_dir)

def masks_gen(images_path, mask_path):
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

    os.makedirs(mask_path, exist_ok=True)

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(images_path, image_file)
        output_path = os.path.join(mask_path, image_file)

        binary_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is None:
            print(f"Error: Failed to load the image '{input_path}'")
            continue

        binary_mask[binary_mask == 0] = 255
        cv2.imwrite(output_path, binary_mask)

        print(f"Modified mask image created successfully: '{output_path}'")
masks_gen(images_dir, processed_img_dir)


