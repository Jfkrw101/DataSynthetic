import glob
import os
import random

NAME_OBJ = "starbucks"
PATH = os.getcwd()
print(f"working on PATH: {PATH}")

def split_dataset(input_path, output_path):
    current_dir = os.path.expanduser(input_path)
    output_dir = os.path.expanduser(output_path)

    percentage_train = 70

    train_file = os.path.join(output_dir, 'train.txt')
    test_file = os.path.join(output_dir, 'test.txt')

    image_paths = glob.glob(os.path.join(current_dir, "*.jpg"))

    random.shuffle(image_paths)

    split_index = int(len(image_paths) * (percentage_train / 100.0))

    train_paths = image_paths[:split_index]
    test_paths = image_paths[split_index:]

    with open(train_file, 'w') as file_train:
        for path in train_paths:
            title, ext = os.path.splitext(os.path.basename(path))
            file_train.write(path + "\n")

    with open(test_file, 'w') as file_test:
        for path in test_paths:
            title, ext = os.path.splitext(os.path.basename(path))
            file_test.write(path + "\n")



split_dataset(PATH + f"/dataset/{NAME_OBJ}/datasets", PATH + f"/dataset/{NAME_OBJ}")
