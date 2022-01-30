import os
from preprocess import apply_blur


def main():
    # PreProcess the images from the source Gopro Large dataset
    # Adding convolutions with PSF
    # abs_path should point where GOPRO_Large_all extracted folder is
    # GOPRO_Large_all/
    #   train/
    #   test/
    abs_path = os.path.abspath(os.path.join(os.curdir, ".."))
    apply_blur(abs_path)

if __name__ == "__main__":
    main()