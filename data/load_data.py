import os

from torch.utils.data import DataLoader

from .loader.getter import DataLoaderGetter
from .loader.params import DataLoaderParams
from data.sub_type import SubType
from data.type import Type


def load_data(path: str, batch_size: int, train: bool, shuffle: bool, limit: float, data_type: str):
    root_path = path
    if data_type == "monkaa":
        type: Type = Type.MONKAA
    else:
        type: Type = Type.FLYING_CHAIRS2
    sub_type_left: SubType = SubType.FUTURE_LEFT
    sub_type_right: SubType = SubType.FUTURE_RIGHT
    data_loader_params: DataLoaderParams = \
        DataLoaderParams(root_path=root_path, type=type, sub_types = [sub_type_left, sub_type_right],
                         batch_size=batch_size, shuffle=shuffle)

    train_loader: DataLoader = DataLoaderGetter.get_by_params(data_loader_params, train=train, limit=limit)

    #it = iter(train_loader)
    #dl_length = len(train_loader)
    #first = next(it)
    #second = next(it)
    #print(first)

    return train_loader


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
    data_dir = "/home/jupyter"

    load_data(data_dir, batch_size=1, train=True, shuffle=False, limit=0.9)
    load_data(data_dir, batch_size=1, train=False, shuffle=False, limit=0.9)

