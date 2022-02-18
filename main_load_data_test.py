import os

from torch.utils.data import DataLoader

from data.loader.getter import DataLoaderGetter
from data.loader.params import DataLoaderParams
from data.loader.sub_type import SubType
from data.type import Type


def load_data(path: str, batch_size: int, train: bool, shuffle: bool, limit: float = 1.0):
    root_path = path
    type: Type = Type.FLYING_CHAIRS2
    sub_type_left: SubType = SubType.NOT_RELEVANT
    sub_type_right: SubType = SubType.NOT_RELEVANT
    data_loader_params: DataLoaderParams = \
        DataLoaderParams(root_path=root_path, type=type, sub_types = [sub_type_left, sub_type_right],
                         batch_size=batch_size, shuffle=shuffle)

    train_loader: DataLoader = DataLoaderGetter.get_by_params(data_loader_params, train=train, limit=limit)

    it = iter(train_loader)
    dl_length = len(train_loader)
    first = next(it)
    second = next(it)
    print(first)

    return train_loader


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
    data_dir = "/home/jupyter"

    load_data(data_dir, batch_size=1, train=True, shuffle=False)
    load_data(data_dir, batch_size=1, train=False, shuffle=False)

