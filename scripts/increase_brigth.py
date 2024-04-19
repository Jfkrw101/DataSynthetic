import cv2
import numpy as np
import os

def increase_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image

name_data = ['apple', 'banana', 'baseball', 'bleach_cleanser', 'bowl', 'chips_can', 'cracker_box', 'dice', 'fork', 'gelatin_box', 'golf_ball', 'knife', 'lemon', 'master_chef_can', 'mini_soccer_ball', 'mug', 'mustard_bottle', 'orange', 'peach', 'pear', 'plate', 'plum', 'potted_meat_can', 'pudding_box', 'racquetball', 'rubiks_cube', 'softball', 'spoon', 'strawberry', 'sugar_box', 'tennis_ball', 'tomato_soup_can', 'tuna_fish_can']

for i, name in enumerate(name_data):
    folder_path = f"/home/jf/setup_train/data/{name}/images"
    out = f'/home/jf/setup_train/data/{name}/brightened'
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)

            brightened_image = increase_brightness(image, 70)

            output_path = os.path.join(out, file.split(".jpg")[0] + ".jpg")
            cv2.imwrite(output_path, brightened_image)

            print(f"Processed {file} Brightened image saved as {output_path}")
