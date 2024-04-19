import cv2
import os

# Set the input image directory and output video directory
image_dir = '/home/jf/setup_train/copy'  # Replace with your image directory path
output_video_dir = '/home/jf/darknet/data'  # Replace with your desired output video directory path

# Create the output video directory if it doesn't exist
os.makedirs(output_video_dir, exist_ok=True)

# Set the output video file name
output_filename = 'valid.mp4'

# Set the duration to display each image in seconds
image_display_duration = 1  # 3 seconds

# Get a list of image file names in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Sort the image files to maintain order
image_files.sort()

# Load the first image to get dimensions
first_image_path = os.path.join(image_dir, image_files[0])
image = cv2.imread(first_image_path)
height, width, layers = image.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
output_path = os.path.join(output_video_dir, output_filename)
out = cv2.VideoWriter(output_path, fourcc, 1 / image_display_duration, (width, height))

# Write the image frames to the video
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    # Repeat each image for the specified duration
    num_frames = int(image_display_duration * 30)  # Assuming 30 frames per second
    for _ in range(num_frames):
        out.write(image)

# Release the VideoWriter
out.release()

print("Video created successfully at:", output_path)
