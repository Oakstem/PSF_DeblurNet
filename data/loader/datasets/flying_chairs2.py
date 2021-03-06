import copy
import os
import os.path
import cv2
import random
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from data.sub_type import SubType
from data.files_reader.flo import read_flo


class FlyingChairs2(Dataset):

    def __init__(self, dataset_path: str, subtypes: [SubType], train: bool, limit_percent: float):

        slash = "\\" if os.name == "nt" else "/"
        files_paths = os.path.join(dataset_path, "train" if train else "val") + slash

        self.subtype: SubType = subtypes
        self.img_size = 256
        self.div_flow = 20
        # In case you want to limit training to a smaller dataset
        files_blurred: [] = None
        files_optical: [] = None
        #files_left = self.get_files(files_paths, "-img_0.png")
        #files_right = self.get_files(files_paths, "-img_1.png")
        files_blurred = self.get_files(files_paths, "-img_blurred.png", limit=limit_percent)
        files_flo = copy.deepcopy(files_blurred)
        for idx, _ in enumerate(files_flo):
            files_flo[idx] = files_flo[idx].replace("-img_blurred.png", "-flow_01.flo")
        #files_flo = self.get_files(files_paths, "-flow_01.flo")
        #files_occ_weights = self.get_files(files_paths, "-occ_weights_01.pfm")
        #files_mb_weights = self.get_files(files_paths, "-mb_weights_01.pfm")
        #files_mb = self.get_files(files_paths, "-mb_01.png")

        self.files_blurred: [] = files_blurred
        self.files_optical: [] = files_flo

    def __len__(self):
        return len(self.files_blurred)

    def __getitem__(self, index):
        """Dataset parameters:
        Train:
        Input: Mean: tensor([0.5872, 0.5912, 0.5727]), Std: tensor([0.2253, 0.2319, 0.2542])
        Target: Mean: tensor([-0.0008, -0.0094]), Std: tensor([0.6536, 0.6616])"""

        image_blurred_path = self.files_blurred[index]
        image_blurred: ndarray = self.load_image(image_blurred_path)

        image_blurred_height = image_blurred.shape[0]
        image_blurred_width = image_blurred.shape[1]
        image_blurred_channels = image_blurred.shape[2]

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(self.img_size), transforms.Normalize(mean=[0.5872, 0.5912, 0.5727], std=[0.2253, 0.2319, 0.2542])])
        image_blurred_tensor: Tensor = transform(image_blurred)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((image_blurred_height, image_blurred_width)),
             transforms.CenterCrop(self.img_size),
             transforms.Normalize(mean=[0, 0], std=[self.div_flow, self.div_flow])])

        image_optical_path = self.files_optical[index]
        image_optical: ndarray = read_flo(self.files_optical[index])[..., :2]
        image_optical_tensor: Tensor = transform(image_optical.copy())

        return image_blurred_tensor, image_optical_tensor, index

    def load_image(self, image_path):
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_files(self, dir, files_extention: str, limit: float):
        files = []
        for f in os.scandir(dir):
            if f.is_file():
                if files_extention in f.name:
                    files.append(f.path)
        random.shuffle(files)
        nb_files = len(files)
        files = files[:int(nb_files*limit)]

        return files
