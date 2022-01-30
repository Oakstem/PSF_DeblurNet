import torch
import numpy as np
import imageio
import os
import re
from interp import get_interpolations
from pathlib import Path
from data.IMPROVE_camera_model import camera_model
from torchvision.transforms import Resize
from torchvision import transforms



def gamma_inv(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, (1/gamm))


def gamma(img: np.ndarray, gamm: float = 2.2):
    return img**(gamm)


def get_interp_nb(flow: np.ndarray, min_nb:int=5, max_pxl_step: int=200, scale_reduct: int=16):
    # Find L2 distance
    dist = np.sqrt(flow[..., 0]**2+flow[..., 1]**2)
    step_sz = np.max(dist)
    if step_sz > max_pxl_step:
        return None
    mx_flow = np.max((step_sz, min_nb))
    return int(np.round(mx_flow/scale_reduct))


def get_dataset_path(data_type: str, root: str):
    if "monkaa" in data_type.lower():
        # Path(os.path.join(root, "\Sampler\Monkaa\optical_flow\PNGs")).mkdir(parents=True, exist_ok=True)
        target_root = os.path.join(root, 'Monkaa/blurred')
        flow_root = os.path.join(root, 'Monkaa/optical_flow')
        rgb_root = os.path.join(root, 'Monkaa/frames_cleanpass')
        print(f"Looking for rgb images at root path:{rgb_root}"
              f"Target blurred will be saved at:{target_root}")

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
    # tmp_img = torch.tensor(readPFM(imgs[0]), device=device)
    tmp_img = torch.tensor([readPFM(im)[0] for im in imgs], device=device).permute(0,3,1,2)
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


def apply_blur(root: str, nb_imgs: int=49, target_sz: list = [270, 480], apply_gamma: bool=True):
    apply_resize = False

    nb_GT_in_batch = 2

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    psfs = torch.load('data/learned_code_27.pt').to(device)
    cam = camera_model(psfs.shape)
    cam.to(device)

    Rsz = Resize(target_sz)

    target_root, flow_root, rgb_root = get_dataset_path("monkaa", root)
    print(f"Target_root:{target_root}")
    if target_root is None:
        return

    imgs_list = get_filenames(rgb_root)
    total_imgs = sum(len(l) for l in imgs_list)
    print(f"Total number of images to process:{total_imgs}")
    res_cnt = 0
    for scn in imgs_list:
        scene_name = scn[0].split("/")[-3]
        print(f"scene_name:{scene_name}")
        Path(os.path.join(target_root, scene_name)).mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(scn[:-nb_GT_in_batch+1]):
            img_pair = scn[idx:idx+nb_GT_in_batch]
            img_name = img.split("/")[-1]
            bl_path = os.path.join(target_root, f"{scene_name}/{img_name}")

            # batch = interp_frames(img_pair[0], img_pair[1], nb_imgs)
            batch = get_interpolations(img_pair[0], img_pair[1], nb_imgs-2, Rsz, apply_gamma=True)

            # Apply PSF convolution to the stacked images
            blr_img = cam(batch, psfs)
            # Apply back the gamma func
            if apply_gamma:
                blr_img = gamma(blr_img)
                # blr_img = blr_img.round()
            blr_img = transforms.ToPILImage()(blr_img).convert("RGB")
            # Save results to /blurred  directory
            blr_img.save(bl_path)

            res_cnt += 1
            if res_cnt%10 == 0:
                print(f"Number of processed images: {res_cnt}/{total_imgs}")


def get_filenames(root):
    res_list = []
    print(f"In:{root}")
    for sub, dirs, files in os.walk(root):
        print(f"searching for imgs at:{root} "
              f"sub:{sub}, dirs:{dirs}")
        if not dirs:
            file_list = [os.path.join(sub, f) for f in files]
            file_list.sort()
            res_list += [file_list]
            # if 'train' in sub:
            #     train_list += [file_list]
            # else:
            #     test_list += [file_list]

    return res_list


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale