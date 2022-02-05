from data.loader.sub_type import SubType
from data.loader.type import Type


class DataLoaderParams:
    def __init__(self, type: Type, sub_types: [SubType], root_path: str,
                 batch_size: int, shuffle: bool = True):
        self.type: Type = type
        self.sub_types: SubType = sub_types
        self.root_path: str = root_path
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
