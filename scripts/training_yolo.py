import os
import argparse
import subprocess
from masks_gen import masks_gen
from ultralytics import YOLO
from split_process import split_dataset
from PIL import Image


parser = argparse.ArgumentParser(description="YOLO Training Script")
parser.add_argument("--data_yaml", type=str, required=True, help="Path to data YAML file")
parser.add_argument("--data_name", type=str, required=True, help="Name Data folder you want for gen")
parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
parser.add_argument("--device", type=int, nargs='+', default=[0], help="List of GPU devices")
parser.add_argument("--number_img",type=str,default='100',help="Number of Dataset want to Gen")
parser.add_argument("--dataset_name",type=str, required=True,help="Enter your dataset folder")
parser.add_argument("--pretrained",type=str, default='yolov8n.pt',help="Enter your Pretrained Model")
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
args = parser.parse_args()

path_dir = os.getcwd()
datasets_dir = f"{path_dir}/dataset/" + args.dataset_name
split_path = f"{path_dir}/dataset/" + args.dataset_name + '/datsets'
split_saved_path = f"{path_dir}/dataset/" + args.dataset_name
data_dir = f"{path_dir}/data/"
images_dir = data_dir + args.data_name  + "/images"
mask_dir = data_dir + args.data_name + "/masks"

def convert_jpg(input,name):
    for idx, filename in enumerate(os.listdir(input)):
        input_path = os.path.join(input, filename)
            
        if not filename.endswith(".jpg"):
            img = Image.open(input_path)

            img = img.convert("RGB")
            new_filename = f"{name}_{idx}.jpg"
            new_path = os.path.join(input, new_filename)
            img.save(new_path, "JPEG")
            print(f"File {filename} converted and renamed to {new_filename}.")
            os.remove(input_path)
            return True
        else:
            return False
        
# def prepare_data(input_path,output_path):
#     masks_gen__path = os.path.join(path_dir, "masks_gen.py")
#     subprocess.run(["python3", masks_gen__path,"--data_name=" + args.dataset_name],check=True )
#     generate_dataset(args.number_img,folder=dataset_dir,split=args.dataset_name)
#     print("Data Gen Done!!!\n")
#     split_dataset(dataset_dir)
#     print("Split Data Done!!!\n")
#     return True


def yolo_train(data_yaml,ep,device,resume):
    model = YOLO('yolov8n.pt') 
    results = model.train(data=path_dir + data_dir + f"/{data_yaml}", epochs=ep, imgsz=416, device=device, batch=64, val=True,resume=resume)
   

if __name__ == "__main__":
    try:
        masks_gen(images_dir,mask_dir)
        # generate_dataset(args.number_img,folder=data_dir,split=datasets_dir) # use sub process
        # print('Data gen Done!!!')
        subprocess.run(['python3',path_dir + '/datagen.py',"--number_img=" + args.number_img,'--dataset_name='+ args.dataset_name],check=True)
        split_dataset(split_path,split_saved_path)
        print('Split dataset Done!!!')
        print(f'test.txt and train.txt stored in:{datasets_dir}\n')
        yolo_train(args.data_yaml, args.epoch, args.device, args.resume)
    except Exception as e:
        print(f'Error occurred training: {e}')
        