from data.loader.sub_type import SubType
from data.loader.type import Type


class DataLoaderParams:
    def __init__(self, type: Type, sub_type: SubType, root_path: str, input_size: tuple,
                 batch_size: int, shuffle: bool = False, device: int or str = 'cpu'):
        self.type: Type = type
        self.sub_type: SubType = sub_type
        self.root_path: str = root_path
        self.input_size: tuple = input_size
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.device: int or str = device