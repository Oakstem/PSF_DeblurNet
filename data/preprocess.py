import torch
import numpy as np
import imageio
import os
from pathlib import Path
from data.IMPROVE_camera_model import camera_model
from torchvision.transforms import Resize


def gamma_inv(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, (1/gamm))


def gamma(img: np.ndarray, gamm: float = 2.2):
    return img**(gamm)


def apply_blur(root: str, nb_imgs = 49, target_sz: list = [357, 637]):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    psfs = torch.load('data/learned_code_27.pt')
    cam = camera_model(psfs.shape)
    cam.to(device)

    Rsz = Resize(target_sz)

    Path(os.path.join(root, "GOPRO_Processed")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, "GOPRO_Processed/train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, "GOPRO_Processed/test")).mkdir(parents=True, exist_ok=True)

    target_root = os.path.join(root, 'GOPRO_Processed')
    source_root = os.path.join(root, 'GOPRO_Large_all')


    train_list, test_list = get_filenames(source_root)
    total_train_imgs = sum(len(l) for l in train_list)
    total_test_imgs = sum(len(l) for l in test_list)

    res_cnt = 0
    for dir in train_list:
        # Create path for blurred & sharp pairs
        type_dir = os.path.dirname(dir[0]).split("\\")[-2]
        dir_name = os.path.dirname(dir[0]).split("\\")[-1]
        bl_path = os.path.join(target_root, f"{type_dir}/{dir_name}", 'blurred')
        sh_path = os.path.join(target_root, f"{type_dir}/{dir_name}",'sharp')

        for idx, img in enumerate(dir[:-nb_imgs]):
            avg_cnt = 0
            while avg_cnt < nb_imgs:
                tmp = imageio.imread(dir[idx+avg_cnt], pilmode='RGB').astype(float)
                tmp_rsz = Rsz(torch.tensor(tmp, device=device).permute(2,0,1))
                if avg_cnt == 0:
                    sharp_left = tmp_rsz.clone().permute(1,2,0)
                elif avg_cnt == nb_imgs-1:
                    sharp_right = tmp_rsz.clone().permute(1, 2, 0)
                aft_gam_inv = gamma_inv(tmp_rsz)
                resized_im = aft_gam_inv.unsqueeze(0)
                # Stack images for PSF convolutions
                if avg_cnt == 0:
                    img_stack = resized_im
                else:
                    img_stack = torch.cat((img_stack, resized_im), 0)
                avg_cnt += 1

            # Apply PSF convolution to the stacked images
            img_stack = img_stack.type(torch.float32)
            psfd_img = cam(img_stack, psfs)
            # Apply back the gamma func
            blr_img = gamma(psfd_img).permute(1, 2, 0)
            blr_img = blr_img.round()

            # Save results to /blurred & /sharp directories
            Path(bl_path).mkdir(parents=True, exist_ok=True)
            Path(sh_path).mkdir(parents=True, exist_ok=True)
            imageio.imwrite(os.path.join(bl_path, f"{res_cnt}.png"), blr_img.type(torch.uint8))
            imageio.imwrite(os.path.join(sh_path, f"{res_cnt}_L.png"), sharp_left.type(torch.uint8))
            imageio.imwrite(os.path.join(sh_path, f"{res_cnt}_R.png"), sharp_right.type(torch.uint8))

            res_cnt += 1
            if res_cnt%10 == 0:
                print(f"Number of processed images: {res_cnt}/{total_train_imgs+total_test_imgs}")


def get_filenames(root):
    train_list = []
    test_list = []
    for sub, dirs, files in os.walk(root):
        if not dirs:
            file_list = [os.path.join(sub, f) for f in files]
            file_list.sort()
            if 'train' in sub:
                train_list += [file_list]
            else:
                test_list += [file_list]

    return train_list, test_list



def _key_check(path, true_key, false_keys):
    path = os.path.join(path, '')
    if path.find(true_key) >= 0:
        for false_key in false_keys:
            if path.find(false_key) >= 0:
                return False

        return True
    else:
        return False
