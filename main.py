import os
import torch
import argparse
from preprocess import apply_blur
from load_data import load_data

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
    data_path = os.path.join(abs_path, "Monkaa")
    train_dataloader = load_data(abs_path, 5)
    apply_blur(abs_path, start_scn_indx=args.start_indx, apply_gamma=args.gamm, target_sz=args.sz)

if __name__ == "__main__":
    main()