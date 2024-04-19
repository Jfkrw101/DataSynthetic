import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from glob import glob

cwd = os.getcwd()
OBJ_NAME = "starbucks"
data_dir = cwd + '/data' + f"/{OBJ_NAME}"
images_dir = data_dir + '/images'
raw_img_path = data_dir + "/raw_image"
mask_dir = data_dir + '/masks'
print(images_dir)
print(mask_dir)

def change_bg(raw_img_path):
    image_files = [f for f in os.listdir(raw_img_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

    os.makedirs(images_dir, exist_ok=True)

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(raw_img_path, image_file)
        output_path = os.path.join(images_dir, image_file)

        binary_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is None:
            print(f"Error: Failed to load the image '{input_path}'")
            continue

        binary_mask[binary_mask == 0] = 255
        cv2.imwrite(output_path, binary_mask)

        img = glob(images_dir + "/.jpeg")
        for i in range(len(img)):
            imgs = cv2.imread(img[i])
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

        print(f"----------Change Background successfully: {output_path}----------")
    return True


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

        binary_mask[binary_mask < 255] = 0
        cv2.imwrite(output_path, binary_mask)

        print(f"Modified mask image created successfully: '{output_path}'")

res = change_bg(raw_img_path)
if res:
    masks_gen(images_dir,mask_dir)
else:
    print("ERROR")


