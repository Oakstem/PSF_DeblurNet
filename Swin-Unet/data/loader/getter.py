import os

from torch.utils.data import DataLoader, Dataset

from datasets.test import Test
from datasets.monkaa import Monkaa
from params import DataLoaderParams
from type import Type


class DataLoaderGetter:

    @staticmethod
    def get_by_params(data_loader_params: DataLoaderParams, train: bool):
        dataset_path = data_loader_params.root_path
        dataset_path = os.path.join(dataset_path, data_loader_params.type.value)

        dataset: Dataset = None
        if data_loader_params.type == Type.TEST:
            dataset: Test = Test(dataset_path, train)
        if data_loader_params.type == Type.MONKAA:
            dataset: Monkaa = Monkaa(dataset_path, data_loader_params.sub_types)

        data_loader: DataLoader = None
        if dataset is not None:
            data_loader: DataLoader = DataLoader(dataset=dataset,
                                                batch_size=data_loader_params.batch_size,
                                                shuffle=data_loader_params.shuffle)

        return data_loader

