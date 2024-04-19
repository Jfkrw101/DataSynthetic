import os
import argparse

def rename_images(directory,new_name):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]  # Add more extensions if needed

    renamed_count = 0
    for image_file in image_files:
        file_name, file_extension = os.path.splitext(image_file)
        if file_extension.lower() in image_extensions:
            new_name = f"{file_name}_{renamed_count}{file_extension}"
            new_path = os.path.join(directory, new_name)
            old_path = os.path.join(directory, image_file)
            
            os.rename(old_path, new_path)
            renamed_count += 1
            print(f"Renamed: {image_file} -> {new_name}")
    
    print(f"Renamed {renamed_count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename an image with a specific name format.")
    parser.add_argument("-new_name", required=True, help="New name for the image")
    cwd = os.getcwd()
    target_directory = cwd + f'/data/images' 
    rename_images(target_directory)
