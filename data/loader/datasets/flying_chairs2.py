import numpy as np
import os
import os.path
import re
import cv2
import skimage.transform as st
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from ..sub_type import SubType
from ...files_reader.pfm import read_pfm


class FlyingChairs2(Dataset):
    BLURRED = "blurred"
    OPTICAL_FLOW = "sm_optical_flow"

    def __init__(self, dataset_path: str, subtypes: [SubType], train: bool):

        slash = "\\" if os.name == "nt" else "/"
        files_paths = os.path.join(dataset_path, "train" if train else "test") + slash

        self.subtype: SubType = subtypes
        self.img_size = 264
        self.div_flow = 20
        # In case you want to limit training to a smaller dataset
        files_blurred: [] = None
        files_optical: [] = None
        files_left = self.get_files(files_paths, "-img_0.png")
        files_right = self.get_files(files_paths, "-img_1.png")
        files_flo = self.get_files(files_paths, "-flow_01.flo")
        files_occ_weights = self.get_files(files_paths, "-occ_weights_01.pfm")
        files_mb_weights = self.get_files(files_paths, "-mb_weights_01.pfm")

        self.files_blurred: [] = files_left
        self.files_optical: [] = files_occ_weights

        print("aa")

    def __len__(self):
        return len(self.files_blurred)

    def __getitem__(self, index):

        image_blurred_path = self.files_blurred[index]
        image_blurred: ndarray = self.load_image(image_blurred_path)

        image_blurred_height = image_blurred.shape[0]
        image_blurred_width = image_blurred.shape[1]
        image_blurred_channels = image_blurred.shape[2]

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(self.img_size)])
        image_blurred_tensor: Tensor = transform(image_blurred)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((image_blurred_height, image_blurred_width)),
             transforms.CenterCrop(self.img_size),
             transforms.Normalize(mean=[0,0],std=[self.div_flow, self.div_flow])])
             # transforms.CenterCrop(image_blurred_height), transforms.Normalize((0, 0), (5, 5))])

        image_optical_path = self.files_optical[index]
        image_optical: ndarray = read_pfm(image_optical_path)[0][..., :2]
        # image_optical = st.resize(image_optical, (image_blurred_height, image_blurred_width))
        image_optical_tensor: Tensor = transform(image_optical.copy())

        return image_blurred_tensor, image_optical_tensor, index

    def load_image(self, image_path):
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_files(self, dir, files_extention: str):
        files = []
        for f in os.scandir(dir):
            if f.is_file():
                if files_extention in f.name:
                    files.append(f.path)

        return files
