import shutil

# Path to the test.txt file containing image paths
input_file_path = '/home/jf/setup_train/dataset/humans/test.txt'

# Destination directory where you want to copy the images
destination_directory = '/home/jf/setup_train/copy'

# Open and read the text file
with open(input_file_path, 'r') as file:
    image_paths = file.readlines()

# Remove any leading/trailing whitespaces and newline characters
image_paths = [path.strip() for path in image_paths]

# Copy each image to the destination directory
for path in image_paths:
    try:
        shutil.copy(path, destination_directory)
        print(f"Image {path} copied successfully.")
    except Exception as e:
        print(f"Error copying {path}: {e}")
