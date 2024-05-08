import glob
import os
import shutil
import random

path_old_dataset = r"D:\archive\val-400\val-400"
path_new_dataset = r"D:\datasets\val-400"

def process_data_1(path_old_dataset, path_new_dataset):
    full_folder_dirs = glob.glob(os.path.join(path_old_dataset, "*"))


    for folder_dir in full_folder_dirs:
        folder_name = os.path.basename(folder_dir)
        new_path = os.path.join(path_new_dataset, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        image_in_folder = glob.glob(os.path.join(folder_dir, "*.JPEG"))
        new_image_list = random.sample(image_in_folder, 5)
        for image in new_image_list:
            shutil.copy(image, new_path)

process_data_1(path_old_dataset, path_new_dataset)
# total_new = glob.glob(os.path.join(path_new_dataset, "0", "*", "*.JPEG"))
# print(len(total_new))


# tmp_list = [1,2,3,4,5,6,7,8,9,10]
# sub_tmp_list = random.sample(tmp_list, 5)
# print(sub_tmp_list)

def process_data_2(path_dataset):
    fulldir = glob.glob(os.path.join(path_dataset, "test", "*.jpg"))
    for dir in fulldir:
        folder_name = os.path.basename(dir).split("_")[0]
        folder_path = os.path.join(path_dataset, folder_name)

        if (not os.path.exists(folder_path)):
            os.makedirs(folder_path)
        shutil.move(dir, folder_path)

# process_data_2(r"D:\datasets\train-400\34")