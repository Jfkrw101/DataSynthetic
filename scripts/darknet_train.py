import os
import sys
import subprocess
import time
import argparse
from glob import glob
import cv2
import matplotlib.pyplot as plt

name_of_computer = os.environ.get('USERNAME')
path_dir = f"/home/{name_of_computer}/setup_train" 
dataset_dir = f"{path_dir}/dataset"
images_dir = f"{path_dir}/images"
obj_name = "starbucks"
random_img_path = f"{dataset_dir}/{obj_name}/datasets"
pretrained_name = "yolov7-tiny.conv.87"
pretrained_path = f"{path_dir}/pretrained/" + pretrained_name

def train_darknet(gpus):
    package_path = os.path.expanduser(path_dir)

    # #preprocess 


    # ## mask gen ##
    # masking_script = os.path.join(package_path, "masks_gen.py")
    # subprocess.run(["python3",masking_script])
    # print(f"--------------- COMPLETE CREATE BINARY MASK ---------------")



    # data gen
    data_gen_script_path = os.path.join(package_path,"datagen.py")
    start_time = time.time()
    subprocess.run(["python3",data_gen_script_path])
    end_time = time.time()
    elaps_time = end_time - start_time
    hours = int(elaps_time // 3600)
    minutes = int((elaps_time % 3600) // 60)
    seconds = int(elaps_time % 60)
    print(f"Datagen execution time: {hours} hrs {minutes} mins {seconds} secs")
    
    img_random = glob(random_img_path + "/*.jpg")
    annotate_txt = glob(random_img_path + "/*.txt")

    

    for i in range(10):
        img_path = img_random[i]
        img_base_name = os.path.splitext(os.path.basename(img_path))[0]
        corresponding_txt = os.path.join(random_img_path, img_base_name + ".txt")

        if corresponding_txt in annotate_txt:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with open(corresponding_txt, 'r') as file:
                annotations = file.readlines()

            for annotation in annotations:
                parts = annotation.strip().split()
                if len(parts) == 5:  
                    class_index, x_center, y_center, width, height = map(float, parts)
                    h, w, _ = img.shape
                    x_min = int((x_center - width / 2) * w)
                    y_min = int((y_center - height / 2) * h)
                    x_max = int((x_center + width / 2) * w)
                    y_max = int((y_center + height / 2) * h)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    cv2.putText(img, "classes_idx:" + str(int(class_index)), (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imshow(f"IMG: {i}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"No annotation file found for {img_base_name}.")


    # process split train and test
    process_script_path = os.path.join(package_path, "split_process.py")
    start_time = time.time()
    subprocess.run(["python3", process_script_path])
    end_time = time.time()
    elaps_time = end_time - start_time
    hours = int(elaps_time // 3600)
    minutes = int((elaps_time % 3600) // 60)
    seconds = int(elaps_time % 60)
    print(f"Split execution time: {hours} hrs {minutes} mins {seconds} secs")
    print("Data generation and Split test.txt,train.txt complete ...\n")


    req_train = input("Ready to train ? (y/n): ")
    if req_train == 'y':
    

        print("Start Training with Darknet ....")
        darknet_path = f"/home/skuba/darknet"

        os.chdir(darknet_path)
        print(f"CHANGE PATH TO {darknet_path}")


        start_time = time.time()
        cmd = f"./darknet detector train {path_dir}/cfg/{obj_name}/{obj_name}.data {path_dir}/cfg/{obj_name}/{obj_name}.cfg {pretrained_path} {gpus} -map"
        os.system(cmd)
        end_time = time.time()
        elaps_time = end_time - start_time
        hours = int(elaps_time // 3600)
        minutes = int((elaps_time % 3600) // 60)
        seconds = int(elaps_time % 60)
        print(f"Execution time: {hours} hrs {minutes} mins {seconds} secs")
        print("Training completed.")
    else:
        print("System not ready for training check your data or dataset")
        sys.exit(1)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Darknet")
    parser.add_argument("-gpus", default="0", help="Choose number of gpu devices to use (e.g., '-gpus 0,1')")
    args = parser.parse_args()

    try:
        train_darknet(args.gpus)
    except Exception as e:
        print("Error occurred during training:", str(e))
        sys.exit(1)
