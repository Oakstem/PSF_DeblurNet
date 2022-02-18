import os
import os.path
import cv2
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from data.sub_type import SubType


class FlyingChairs2(Dataset):

    def __init__(self, dataset_path: str, subtypes: [SubType], train: bool):

        slash = "\\" if os.name == "nt" else "/"
        files_paths = os.path.join(dataset_path, "train" if train else "val") + slash

        self.subtype: SubType = subtypes
        self.img_size = 264
        self.div_flow = 20
        # In case you want to limit training to a smaller dataset
        files_blurred: [] = None
        files_optical: [] = None
        files_left = self.get_files(files_paths, "-img_0.png")
        files_right = self.get_files(files_paths, "-img_1.png")
        files_blurred = self.get_files(files_paths, "-img_blurred.png")
        files_flo = self.get_files(files_paths, "-flow_01.flo")
        files_occ_weights = self.get_files(files_paths, "-occ_weights_01.pfm")
        files_mb_weights = self.get_files(files_paths, "-mb_weights_01.pfm")
        files_mb = self.get_files(files_paths, "-mb_01.png")

        self.files_blurred: [] = files_left
        self.files_optical: [] = files_mb

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

        image_optical_path = self.files_optical[index]
        image_optical: ndarray = self.load_image(image_optical_path)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(self.img_size)])
        image_optical_tensor: Tensor = transform(image_optical)

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
