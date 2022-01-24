import os

from torch.utils.data import DataLoader, Dataset

from data.loader.flow_dataset import FlowDataset
from data.loader.params import DataLoaderParams


class DataLoaderGetter:

    @staticmethod
    def get_by_params(data_loader_params: DataLoaderParams, train: bool):
        dataset_path = data_loader_params.root_path
        if data_loader_params.dataset_name != "":
            dataset_path = os.path.join(dataset_path, data_loader_params.dataset_name)

        dataset: FlowDataset = FlowDataset(dataset_path, train, data_loader_params.input_size)

        data_loader: DataLoader = DataLoader(dataset=dataset,
                                             batch_size=data_loader_params.batch_size,
                                             shuffle=data_loader_params.shuffle)

        return data_loader

