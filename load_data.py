from torch.utils.data import DataLoader

from data.loader.getter import DataLoaderGetter
from data.loader.params import DataLoaderParams
from data.loader.sub_type import SubType
from data.loader.type import Type


def load_data():
    root_path = "data/"
    type: Type = Type.MONKAA
    sub_type: SubType = SubType.FUTURE_LEFT
    data_loader_params: DataLoaderParams = \
        DataLoaderParams(root_path=root_path, type=type, sub_type = sub_type, input_size=(640, 480),
                         batch_size=1, shuffle=False, device='cpu')

    train_loader: DataLoader = DataLoaderGetter.get_by_params(data_loader_params, train=True)

    it = iter(train_loader)
    first = next(it)
    second = next(it)
    print("dd")


if __name__ == "__main__":
    load_data()

