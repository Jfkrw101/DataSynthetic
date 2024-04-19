import os
from PIL import Image
import cv2

def convert_to_yolo_format(class_index, box, image_width, image_height):
    x, y, width, height = box
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return f"{class_index} {x_center} {y_center} {normalized_width} {normalized_height}"

name_data = ['apple', 'banana', 'baseball', 'bleach_cleanser', 'bowl', 'chips_can', 'cracker_box', 'dice', 'fork', 'gelatin_box', 'golf_ball', 'knife', 'lemon', 'master_chef_can', 'mini_soccer_ball', 'mug', 'mustard_bottle', 'orange', 'peach', 'pear', 'plate', 'plum', 'potted_meat_can', 'pudding_box', 'racquetball', 'rubiks_cube', 'softball', 'spoon', 'strawberry', 'sugar_box', 'tennis_ball', 'tomato_soup_can', 'tuna_fish_can']

for i, name in enumerate(name_data):

    image_dir = f"/home/jf/setup_train/data/{name}/brightened"
    mask_dir = f"/home/jf/setup_train/data/{name}/masks_brightened"
    output_dir = f"/home/jf/setup_train/data/{name}/annotated"

    class_index = i

    image_files = os.listdir(image_dir)

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask_file = image_file.replace(".jpg", ".pbm")
        mask_path = os.path.join(mask_dir, mask_file)
        print(image_path)
        
        output_file = image_file.replace(".jpg", ".txt")
        output_path = os.path.join(output_dir, output_file)

        if os.path.exists(mask_path):
            image = cv2.imread(image_path)
            print(mask_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.medianBlur(mask, 15)

            image_height, image_width, _ = image.shape
            mask_height, mask_width = mask.shape
            
            image_resize = cv2.resize(image, [int(mask_height/4),int(mask_width/4)])
            _, binary_mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)
            mask_resize = cv2.resize(binary_mask, [int(mask_height/4),int(mask_width/4)])
            # cv2.imshow("original", image_resize)
            # cv2.imshow("mask", mask_resize)
            # cv2.waitKey(0)
            
            if image_width != mask_width or image_height != mask_height:
                print(f"Dimensions mismatch for image: {image_file}")
                continue

            with open(output_path, "w") as f:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    print(len(contours))
                    x, y, w, h = cv2.boundingRect(contour)
                    # print(x, y, w, h)
                    box = (x, y, w, h)  # Create a bounding box around the contour
                    yolo_annotation = convert_to_yolo_format(class_index, box, image_width, image_height)
                    f.write(yolo_annotation + "\n")
                    
        else:
            print(f"Mask not found for image: {image_file}")