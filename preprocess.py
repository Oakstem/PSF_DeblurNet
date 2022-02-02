import time
import torch
import numpy as np
import os
import re
from pathlib import Path
from data.IMPROVE_camera_model import camera_model
from torchvision.transforms import Resize
from torchvision import transforms

from utils import read_pfm

if torch.cuda.is_available():
    # from interp import get_interpolations
    from interpolations.interpolations import get_interpolations


def gamma_inv(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, (1/gamm))


def gamma(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, gamm)


def get_interp_nb(flow_root: str, scene_name: str, img_name: str, min_nb:int=5, max_pxl_step: int=100, scale_reduct: int=2):
    # Find L2 distance of flow movement, if within threshold, return number of frame interpolation
    of_path = os.path.join(flow_root, scene_name, "into_future/left/OpticalFlowIntoFuture_" +
                           img_name[:-4] + "_L.pfm")
    of = read_pfm(of_path)[0]

    dist = np.sqrt(of[..., 0]**2+of[..., 1]**2)
    step_sz = np.max(dist)

    # Temporary threshold:
    if not 170 > step_sz > 100:
        return None, None
    # if step_sz > max_pxl_step:
    #     return None, None
    mx_flow = np.max((step_sz, min_nb))/scale_reduct
    if mx_flow <= 37:
        return 23, mx_flow
    else:
        return 47, mx_flow


def get_dataset_path(data_type: str, root: str):
    if "monkaa" in data_type.lower():
        # Path(os.path.join(root, "\Sampler\Monkaa\optical_flow\PNGs")).mkdir(parents=True, exist_ok=True)
        target_root = os.path.join(root, 'Monkaa/blurred')
        flow_root = os.path.join(root, 'Monkaa/optical_flow')
        rgb_root = os.path.join(root, 'Monkaa/frames_cleanpass')
        Path(target_root).mkdir(parents=True, exist_ok=True)

    elif "gopro" in data_type.lower():
        Path(os.path.join(root, "GOPRO_Processed")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "GOPRO_Processed/train")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(root, "GOPRO_Processed/test")).mkdir(parents=True, exist_ok=True)

        target_root = os.path.join(root, 'GOPRO_Processed')
        source_root = os.path.join(root, 'GOPRO_Large_all')

    else:
        print("Error! Unknown dataset type")
        target_root = ""
        source_root = ""
        flow_root = ""
        rgb_root = ""

    return target_root, flow_root, rgb_root


def interp_frames(i1, i2, nb_interp):
    return torch.tensor(np.ones((nb_interp,3,600,700))).double()


def find_flow(imgs, device):
    # tmp_img = torch.tensor(read_pfm(imgs[0]), device=device)
    tmp_img = torch.tensor([read_pfm(im)[0] for im in imgs], device=device).permute(0,3,1,2)
    xvec = torch.tensor(np.arange(0, tmp_img[0].shape[2]), device=device)
    yvec = torch.tensor(np.arange(0, tmp_img[0].shape[1]), device=device)
    x_coord, y_coord = np.meshgrid(xvec, yvec)
    x_coord = torch.tensor(x_coord, device=device).long()
    y_coord = torch.tensor(y_coord, device=device).long()
    for im in tmp_img:
        x_coord_new = torch.add(x_coord, im[0][y_coord, x_coord])
        y_coord_new = torch.add(y_coord, im[1][y_coord, x_coord])
        x_coord = x_coord_new
        y_coord = y_coord_new
    return x_coord
    # for img in imgs:


def load_cam_models(device: torch.device):
    psf49 = torch.load('data/learned_code_27.pt').to(device)
    psf25 = torch.load('data/learned_code_S25_27.pt').to(device)
    cam49 = camera_model(psf49.shape).to(device)
    cam25 = camera_model(psf25.shape).to(device)
    return [cam25, cam49], [psf25, psf49]


def create_scene_dir(target_root: str, filename: str, idx: int):
    # Windows / Linux environment different paths
    if len(filename.split("/")) > len(filename.split("\\")):
        scene_name = filename.split("/")[-3]
    else:
        scene_name = filename.split("\\")[-3]
    print(f"scene_name:{scene_name}, index:{idx}")
    Path(os.path.join(target_root, scene_name)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(target_root, f"{scene_name}/left")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(target_root, f"{scene_name}/right")).mkdir(parents=True, exist_ok=True)
    return scene_name


def get_blurred_img_path(target_root: str, scene_name: str, filename: str, side: str):
    if len(filename.split("/")) > len(filename.split("\\")):
        img_name = filename.split("/")[-1]
    else:
        img_name = filename.split("\\")[-1]
    blr_path = os.path.join(target_root, f"{scene_name}/{side}/{img_name}")
    return blr_path, img_name


def apply_psf(cam: list, psfs: list, batch: torch.Tensor, nb_imgs: int, apply_gamma: bool=True):
    # Apply PSF convolution to the stacked images
    if nb_imgs == 23:
        blr_img = cam[0](batch, psfs[0])
    else:
        blr_img = cam[1](batch, psfs[1])
    if apply_gamma:
        blr_img = gamma(blr_img)
    # Apply back the gamma func
    blr_img = transforms.ToPILImage()(blr_img).convert("RGB")

        # blr_img = blr_img.round()

    return blr_img


def apply_blur(root: str, start_scn_indx: int=0,
               target_sz: list = [270, 480], apply_gamma: bool=True, side: str ="right"):
    apply_resize = False

    nb_GT_in_batch = 2

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cams, psfs = load_cam_models(device)

    Rsz = Resize(target_sz)

    target_root, flow_root, rgb_root = get_dataset_path("monkaa", root)
    print(f"Target_root:{target_root}")
    if target_root is None:
        return

    imgs_list, total_imgs = get_filenames(rgb_root)

    res_cnt = 0
    cnt_prev = 0
    t = time.time()
    max_flow_list = []
    for idx in range(start_scn_indx, len(imgs_list)):
        scene_name = create_scene_dir(target_root, imgs_list[idx][0], idx)

        # Loop through all images in the scene
        for idy, img in enumerate(imgs_list[idx][:-nb_GT_in_batch+1]):
            img_pair = imgs_list[idx][idy:idy+nb_GT_in_batch]

            # Get blurred img path
            blr_path, img_name = get_blurred_img_path(target_root, scene_name, img, side)

            # Get the OF
            nb_imgs, mx_flow = get_interp_nb(flow_root, scene_name, img_name)
            max_flow_list.append(mx_flow)

            if nb_imgs is not None and len(img_pair) == 2:
                if device == torch.device("cuda"):
                    batch = get_interpolations(img_pair[0], img_pair[1], nb_imgs, Rsz, apply_gamma=True)
                else:
                    # only for debugging, not a real interpolation
                    batch = interp_frames(img_pair[0], img_pair[1], nb_imgs+2)

                # Apply the suitable PSF conv
                blr_img = apply_psf(cams, psfs, batch, nb_imgs, apply_gamma)
                # Save results to /blurred  directory
                blr_img.save(blr_path)

                res_cnt += 1
                if res_cnt%10 == 0:
                    elapsed = time.time() - t
                    print(f"Number of processed images: {res_cnt}/{total_imgs}, scene index:{idx}, time per image:{(elapsed/(res_cnt-cnt_prev)):0.1f} sec")
                    cnt_prev = res_cnt
                    t = time.time()


def get_filenames(root):
    res_list = []
    total_imgs = 0
    for sub, dirs, files in os.walk(root):
        # Discard the "right" dirs since no stereo is needed
        if not dirs and "right" not in sub:
            file_list = [os.path.join(sub, f) for f in files]
            file_list.sort()
            res_list += [file_list]
            # if 'train' in sub:
            #     train_list += [file_list]
            # else:
            #     test_list += [file_list]
            total_imgs+= len(file_list)
    print(f"Total number of images to process:{total_imgs}")

    return res_list, total_imgs


def move_2_left_dir(root):
    for sub, dirs, files in os.walk(root):
        # print(f"sub:{sub}, dirs:{dirs}, files:{files}")
        if not dirs:
            left_path = os.path.join(sub, "left")
            # right_path = os.path.join(sub, "right")
            Path(left_path).mkdir(parents=True, exist_ok=True)
            # Path(right_path).mkdir(parents=True, exist_ok=True)
            for file in files:
                filepath = os.path.join(sub, file)
                os.rename(filepath, os.path.join(left_path, file))
