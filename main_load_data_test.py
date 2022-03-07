import os

import torch
from torch.utils.data import DataLoader

from data.loader.getter import DataLoaderGetter
from data.loader.params import DataLoaderParams
from data.sub_type import SubType
from data.type import Type


def load_data(path: str, batch_size: int, train: bool, shuffle: bool, limit: float = 1.0):
    root_path = path
    type: Type = Type.FLYING_CHAIRS2
    sub_type_left: SubType = SubType.NOT_RELEVANT
    sub_type_right: SubType = SubType.NOT_RELEVANT
    data_loader_params: DataLoaderParams = \
        DataLoaderParams(root_path=root_path, type=type, sub_types = [sub_type_left, sub_type_right],
                         batch_size=batch_size, shuffle=shuffle)

    train_loader: DataLoader = DataLoaderGetter.get_by_params(data_loader_params, train=True, limit=limit)
    input_mean, input_std, target_mean, target_std = get_mean_and_std(train_loader)
    print("Input: Mean: " + str(input_mean) + ", Std: " + str(input_std))
    print("Target: Mean: " + str(target_mean) + ", Std: " + str(target_std))

    it = iter(train_loader)
    dl_length = len(train_loader)
    first = next(it)
    second = next(it)
    print(first)

    return train_loader


def get_mean_and_std(dataloader):
    input_channels_sum, input_channels_squared_sum, target_channels_sum, target_channels_squared_sum, num_batches = 0, 0, 0, 0, 0
    for data in dataloader:
        # Mean over batch, height and width, but not over the channels
        input, target, index = data
        input_channels_sum += torch.mean(input, dim=[0, 2, 3])
        input_channels_squared_sum += torch.mean(input ** 2, dim=[0, 2, 3])
        target_channels_sum += torch.mean(target, dim=[0, 2, 3])
        target_channels_squared_sum += torch.mean(target ** 2, dim=[0, 2, 3])
        num_batches += 1

    input_mean = input_channels_sum / num_batches
    target_mean = target_channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    input_std = (input_channels_squared_sum / num_batches - input_mean ** 2) ** 0.5
    target_std = (target_channels_squared_sum / num_batches - target_mean ** 2) ** 0.5

    return input_mean, input_std, target_mean, target_std


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
    #data_dir = "/Users/mishka/PycharmProjects/PSF_DeblurNet/data/loader/datasets"
    data_dir = "/home/jupyter"

    load_data(data_dir, batch_size=32, train=True, shuffle=False)
    #load_data(data_dir, batch_size=1, train=False, shuffle=False)

