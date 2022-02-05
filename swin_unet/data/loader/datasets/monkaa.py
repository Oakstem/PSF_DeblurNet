import numpy as np
import os
import os.path

import cv2
import skimage.transform as st
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from ..sub_type import SubType
from utils import read_pfm


class Monkaa(Dataset):
    BLURRED = "blurred"
    OPTICAL_FLOW = "optical_flow"

    def __init__(self, dataset_path: str, subtypes: [SubType]):

        # self.file_path = os.path.join(dataset_path, "train" if train else "test")
        self.file_path: str = dataset_path

        self.subtype: SubType = subtypes
        self.img_size = 224

        slash = "\\" if os.name == "nt" else "/"
        files_blurred: [] = None
        files_optical: [] = None
        _, files_blurred = self.run_fast_scandir(self.file_path + slash + self.BLURRED, [".png"])
        _, files_optical = self.run_fast_scandir(self.file_path + slash + self.OPTICAL_FLOW, [".pfm"])

        files_blurred_filtered: [] = []
        files_optical_filtered: [] = []
        if SubType.FUTURE_LEFT in subtypes:
            files_blurred_filtered += [k for k in files_blurred if 'left' in k]
            files_optical_filtered += [k for k in files_optical if 'left' in k and 'into_future' in k]
        if SubType.FUTURE_RIGHT in subtypes:
            files_blurred_filtered += [k for k in files_blurred if 'right' in k]
            files_optical_filtered += [k for k in files_optical if 'right' in k and 'into_future' in k]
        if SubType.PAST_LEFT in subtypes:
            files_blurred_filtered.append([k for k in files_blurred if 'left' in k])
            files_optical_filtered.append([k for k in files_optical if 'left' in k and 'into_past' in k])
        if SubType.PAST_RIGHT in subtypes:
            files_blurred_filtered.append([k for k in files_blurred if 'right' in k])
            files_optical_filtered.append([k for k in files_optical if 'right' in k and 'into_past' in k])

        files_optical_dict = {}
        for file_optical in files_optical_filtered:
            files_optical_dict[file_optical] = file_optical

        files_blurred_dict = {}
        file_blurred: str = ""
        camera_time: str = "into_future"
        for file_blurred in files_blurred_filtered:
            file_blurred_path_remainder: str = file_blurred.replace(self.file_path + slash + self.BLURRED + slash, "")
            file_blurred_path_remainder_list = file_blurred_path_remainder.split(slash)
            scene_name: str = file_blurred_path_remainder_list[0]
            if file_blurred_path_remainder_list[1] == "into_future" or file_blurred_path_remainder_list[1] == "into_past":
                camera_time: str = file_blurred_path_remainder_list[1]
                camera_side: str = file_blurred_path_remainder_list[2]
                camera_side_first_letter: str = camera_side[0].upper()
                file_name: str = file_blurred_path_remainder_list[3]
            else:
                camera_side: str = file_blurred_path_remainder_list[1]
                camera_side_first_letter: str = camera_side[0].upper()
                file_name: str = file_blurred_path_remainder_list[2]
            file_name_no_extention: str = file_name.split(".")[0]

            file_optical_path_to_search = self.file_path + slash + \
                                          self.OPTICAL_FLOW + slash + \
                                          scene_name + slash + \
                                          camera_time + slash + \
                                          camera_side + slash + \
                                          "OpticalFlowIntoFuture_" + \
                                          file_name_no_extention + "_" + \
                                          camera_side_first_letter + ".pfm"

            if file_optical_path_to_search in files_optical_dict:
                files_blurred_dict[file_blurred] = file_optical_path_to_search

        self.files_blurred: [] = list(files_blurred_dict.keys())
        self.files_optical: [] = list(files_blurred_dict.values())

        # print("aa")

    def __len__(self):
        return len(self.files_blurred)

    def __getitem__(self, index):

        image_blurred_path = self.files_blurred[index]
        image_blurred: ndarray = self.load_image(image_blurred_path)

        image_blurred_height = image_blurred.shape[0]
        image_blurred_width = image_blurred.shape[1]
        image_blurred_channels = image_blurred.shape[2]

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(image_blurred_height), transforms.Resize((self.img_size))])
        image_blurred_tensor: Tensor = transform(image_blurred)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((image_blurred_height, image_blurred_width)),
             transforms.CenterCrop(image_blurred_height), transforms.Resize(self.img_size)])

        image_optical_path = self.files_optical[index]
        image_optical: ndarray = read_pfm(image_optical_path)[0][..., :2]
        image_optical = st.resize(image_optical, (image_blurred_height, image_blurred_width))
        image_optical_tensor: Tensor = transform(image_optical)

        return image_blurred_tensor, image_optical_tensor

    def load_image(self, image_path):
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def run_fast_scandir(self, dir, files_extentions):
        subfolders, files = [], []
        folder_files_num = 0
        folder_files = []
        for f in os.scandir(dir):
            if f.is_dir():
                subfolders.append(f.path)
            if f.is_file():
                filename, file_extension = os.path.splitext(f.name)
                if file_extension.lower() in files_extentions:
                    folder_files.append(f.path)
                    folder_files_num = folder_files_num + 1

        for dir in list(subfolders):
            subfolders_folders, subfolders_files = self.run_fast_scandir(dir, files_extentions)
            subfolders.extend(subfolders_folders)
            files.extend(subfolders_files)

        files.extend(folder_files)

        return subfolders, files



