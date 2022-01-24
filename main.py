import os

from torch.utils.data import DataLoader

from data.loader.getter import DataLoaderGetter
from data.loader.params import DataLoaderParams
from data.preprocess import apply_blur


def main():
    # PreProcess the images from the source Gopro Large dataset
    # Adding convolutions with PSF
    # abs_path should point where GOPRO_Large_all extracted folder is
    # GOPRO_Large_all/
    #   train/
    #   test/
    abs_path = os.path.abspath(os.path.join(os.curdir, "../deep-learning-flower-identifier/"))
    apply_blur(abs_path)

def load_data():
    root_path = "/Users/globalbit/Downloads/uclOpticalFlow_v1.2/scenes"
    data_loader_params: DataLoaderParams = DataLoaderParams(root_path=root_path, dataset_name="",
                                                            input_size=(640, 480), batch_size=1, shuffle=False,
                                                            device='cpu')

    train_loader: DataLoader = DataLoaderGetter.get_by_params(data_loader_params, train=True)

    it = iter(train_loader)
    first = next(it)
    second = next(it)
    print("dd")


if __name__ == "__main__":
    load_data()
    main()

