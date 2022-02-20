import os
import torch
import argparse

from data.load_data import load_data
from data.preprocess.preprocess import apply_blur
from data.sub_type import SubType
from data.type import Type


def main():
    # PreProcess the images from the source Gopro Large dataset
    # Adding convolutions with PSF
    # abs_path should point where GOPRO_Large_all extracted folder is
    # GOPRO_Large_all/
    #   train/
    #   test/
    parser = argparse.ArgumentParser(description='Applying Blur effect with PSF encoding')
    parser.add_argument('--start_indx', '-si', default=0, type=int, help='scene index to start with,'
                                                                         'choose between 0-24 for Monkaa')
    parser.add_argument('--gamm', '-g', action='store_true', help='Choose whether to apply gamma')
    parser.add_argument('--sz', default=[270, 480], type=list, help='Target image size')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    abs_path = os.path.abspath(os.path.join(os.curdir, ".."))
    abs_path = "/home/jupyter/"
    type: Type = Type.FLYING_CHAIRS2
    sub_type: SubType = SubType.NOT_RELEVANT
    #train_dataloader = load_data(abs_path, batch_size=5, train=True, shuffle=False, limit=0.9)
    #test_dataloader = load_data(abs_path, batch_size=5, train=False, shuffle=False, limit=0.9)

    apply_blur(type, sub_type, abs_path, start_scene_index=args.start_indx, target_size=args.sz,
               do_apply_gamma=args.gamm)


if __name__ == "__main__":
    main()