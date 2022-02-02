import numpy as np
import os.path

import cv2
import torch
from torch.utils.data import Dataset


class Test(Dataset):
    def __init__(self, dataset_path: str, train: bool, image_size: tuple ):

        self.train = train
        # self.file_path = os.path.join(dataset_path, "train" if train else "test")
        self.file_path = dataset_path
        #self.file_path = os.path.join(self.file_path, '**/*')

        self.image_size: tuple = image_size

        self.sub_folders, self.files = self.run_fast_scandir(self.file_path, ["1.png", "2.png"])

        # self.samplec = len(subfolders)

    def __len__(self):
        return len(self.sub_folders)

    def __getitem__(self, index):

        sub_folder = self.sub_folders[index]
        image1 = self.load_image(os.path.join(sub_folder, "1.png"))
        image2 = self.load_image(os.path.join(sub_folder, "2.png"))

        image1 = self.normalize_in_place(image1)
        image2 = self.normalize_in_place(image2)

        return torch.tensor(image1, dtype=torch.float32).permute(2,0,1),\
               torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)

    def load_image(self, image_path):
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def run_fast_scandir(self, dir, files_names):
        subfolders, files = [], []
        folder_files_num = 0
        folder_files = []
        for f in os.scandir(dir):
            if f.is_dir():
                subfolders.append(f.path)
            if f.is_file():
                if f.name.lower() in files_names:
                    folder_files.append(f.path)
                    folder_files_num = folder_files_num + 1

        for dir in list(subfolders):
            subfolders_folders, subfolders_files = self.run_fast_scandir(dir, files_names)
            subfolders.extend(subfolders_folders)
            files.extend(subfolders_files)

        if folder_files_num == len(files_names):
            files.extend(folder_files)

        return subfolders, files

    @staticmethod
    def normalize_in_place(x):
        return np.array(np.array(x, dtype=np.float32), dtype=np.float32) / 128.0 - 1.0



