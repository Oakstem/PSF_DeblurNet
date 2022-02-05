import numpy as np
import os.path

import cv2
import skimage.transform as st
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from data.loader.sub_type import SubType
from utils import read_pfm
import torch.nn.functional as nnf



class Monkaa(Dataset):
    def __init__(self, dataset_path: str, subtype: SubType):

        # self.file_path = os.path.join(dataset_path, "train" if train else "test")
        self.file_path: str = dataset_path
        self.subtype: SubType = subtype
        self.img_size = 224

        self.files_blurred: [] = None
        self.files_optical: [] = None
        _, self.files_blurred = self.run_fast_scandir(self.file_path + "/blurred", [".png"])
        _, self.files_optical = self.run_fast_scandir(self.file_path + "/optical_flow", [".pfm"])


        if subtype == SubType.FUTURE_LEFT:
            self.files_blurred = [k for k in self.files_blurred if 'left' in k]
            self.files_optical = [k for k in self.files_optical if 'left' in k and 'into_future' in k]
        if subtype == SubType.FUTURE_RIGHT:
            self.files_blurred = [k for k in self.files_blurred if 'right' in k]
            self.files_optical = [k for k in self.files_optical if 'right' in k and 'into_future' in k]
        if subtype == SubType.PAST_LEFT:
            self.files_blurred = [k for k in self.files_blurred if 'left' in k]
            self.files_optical = [k for k in self.files_optical if 'left' in k and 'into_past' in k]
        if subtype == SubType.Past_RIGHT:
            self.files_blurred = [k for k in self.files_blurred if 'right' in k]
            self.files_optical = [k for k in self.files_optical if 'right' in k and 'into_past' in k]

        # self.samplec = len(subfolders)

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



        #from PIL import Image
        #im = Image.fromarray(image_blurred)
        #im.save("data/loader/datasets/your_file1.png")
        # image_blurred = self.normalize_in_place(image_blurred)
        # im = Image.fromarray(image_blurred)
        # im.save("data/loader/datasets/your_file2.png")

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((image_blurred_height, image_blurred_width)),
             transforms.CenterCrop(image_blurred_height), transforms.Resize(self.img_size)])

        image_optical_path = self.files_optical[index]
        image_optical: ndarray = read_pfm(image_optical_path)[0][..., :2]
        image_optical = st.resize(image_optical, (image_blurred_height, image_blurred_width))
        # image_optical = Image.fromarray((image_optical * 255).astype(np.uint8))
        # image_optical_tensor: Tensor = transforms.ToTensor()(image_optical).unsqueeze_(0)
        image_optical_tensor: Tensor = transform(image_optical)


        # save_image(image_optical_tensor, "data/loader/datasets/your_file4.png")
        #image_blurred_tensor: Tensor = torch.tensor(image_blurred, dtype=torch.float32).permute(2, 0, 1)
        #save_image(image_blurred_tensor, "data/loader/datasets/your_file2.png")
        #tmp_img = torch.tensor(read_pfm(image_optical_path)[0], device='cpu').permute(2, 0, 1)
        #save_image(tmp_img, "data/loader/datasets/your_file5.png")

        # im = Image.fromarray(image_optical)
        # im.save("data/loader/datasets/your_file3.png")
        #
        # image_optical = np.random.random_sample(image_optical.shape) * 255
        # image_optical = image_optical.astype(np.float)
        # #im = Image.fromarray(image_optical)
        # #im.save("data/loader/datasets/your_file3.png")
        #
        # image_optical_tensor: Tensor = transform_to_tensor(image_optical)
        # transform_resize = transforms.Resize(image_blurred_width, image_blurred_height)
        # image_optical_tensor = transform_resize(image_blurred_tensor)
        # #save_image(image_optical_tensor, "data/loader/datasets/your_file4.png")
        # # image_optical = cv2.resize(image_optical, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
        # #image_optical = self.normalize_in_place(image_optical)
        # #image_optical_tensor: Tensor = torch.tensor(image_optical, dtype=torch.float32).permute(2, 0, 1)
        # #out = nnf.interpolate(image_optical_tensor, size=(image_blurred_width, image_blurred_height), mode='bicubic', align_corners=False)
        # #save_image(im, 'data/datasets/im_name.png')

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

        # if folder_files_num == len(files):
        files.extend(folder_files)

        return subfolders, files

    @staticmethod
    def normalize_in_place(x):
        return np.array(np.array(x, dtype=np.float32), dtype=np.float32) / 128.0 - 1.0



