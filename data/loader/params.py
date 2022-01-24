
class DataLoaderParams:
    def __init__(self, root_path: str, dataset_name: str, input_size: tuple, batch_size: int, shuffle: bool = False,
                 device: int or str = 'cpu'):
        self.root_path: str = root_path
        self.dataset_name: str = dataset_name
        self.input_size: tuple = input_size
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.device: int or str = device