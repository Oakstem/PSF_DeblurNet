import time
import torch
import numpy as np
import os

from torch import Tensor

from data.IMPROVE_camera_model import camera_model
from torchvision.transforms import Resize
from torchvision import transforms

from data.files_reader.pfm import read_pfm
from data.preprocess.debug import debug_interp_frames
from data.preprocess.gamma import apply_gamma
from data.preprocess.interpolations.interpolations import get_interpolations
from data.preprocess.files_folders import get_dataset_path, get_blurred_image_path, create_scene_dir, \
    get_filenames_from_subfolders, get_filenames_by_extention
from data.sub_type import SubType
from data.type import Type

NUM_GT_IN_BATCH = 2


def apply_blur(type: Type, sub_type: SubType, data_path: str, start_scene_index: int = 0,
               target_size: list = [270, 480], do_apply_gamma: bool = True):
    target_root, flow_root, rgb_root = get_dataset_path(type, data_path)
    print(f"Target_root:{target_root}")
    if target_root is None:
        return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cams: [camera_model] = None
    psfs: [Tensor] = None
    cams, psfs = load_cam_models(device)

    resize = Resize(target_size)

    if type == Type.MONKAA:
        images_list: [] = []
        total_images_num: int = 0
        side: str == ""
        if sub_type == SubType.PAST_LEFT or sub_type == SubType.FUTURE_LEFT:
            side = "left"
        else:
            side = 'right'
        images_list, total_images_num = get_filenames_from_subfolders(rgb_root, side)
        _apply_blur_monkaa(target_root, flow_root, images_list, total_images_num, start_scene_index, resize,
                           do_apply_gamma, side, device, cams, psfs)
    if type == Type.FLYING_CHAIRS2:
        files_left = get_filenames_by_extention(rgb_root, "-img_0.png")
        files_right = get_filenames_by_extention(rgb_root, "-img_1.png")


def _apply_blur_monkaa(target_root: str, flow_root: str, images_list: [], total_images_num: int, start_scene_index: int,
                       resize: Resize, do_apply_gamma: bool, side: str, device: int or str, cams: [camera_model],
                       psfs: [Tensor]):
    processed_count = 0
    prev_processed_count = 0
    t = time.time()
    for idx in range(start_scene_index, len(images_list)):
        scene_name = create_scene_dir(target_root, images_list[idx][0], idx)

        # Loop through all images in the scene
        for idy, img in enumerate(images_list[idx][:-NUM_GT_IN_BATCH + 1]):
            img_pair = images_list[idx][idy:idy + NUM_GT_IN_BATCH]

            # Get blurred img path
            blur_path, image_name = get_blurred_image_path(target_root, scene_name, img, side)

            # Get the OF
            num_images, _ = get_interpolations_num(flow_root, scene_name, image_name, side)

            if num_images is not None and len(img_pair) == 2:
                if device == torch.device("cuda"):
                    batch = get_interpolations(img_pair[0], img_pair[1], num_images, resize, apply_gamma=True)
                else:
                    # only for debugging, not a real interpolation
                    batch = debug_interp_frames(img_pair[0], img_pair[1], num_images + 2)

                # Apply the suitable PSF conv
                blurred_image = apply_psf(cams, psfs, batch, num_images, do_apply_gamma)
                # Save results to /blurred  directory
                blurred_image.save(blur_path)

                processed_count += 1
                if processed_count % 10 == 0:
                    elapsed = time.time() - t
                    print(f"Number of processed images: {processed_count}/{total_images_num}, scene index:{idx}, "
                          f"time per image:{(elapsed / (processed_count - prev_processed_count)):0.1f} sec")
                    prev_processed_count = processed_count
                    t = time.time()


def get_interpolations_num(flow_root: str, scene_name: str, img_name: str, side: str, min_nb: int = 5,
                           max_pxl_step: int = 170, scale_reduct: int = 2):
    if side == "left":
        letter = "L"
    else:
        letter = "R"
        
    # Find L2 distance of flow movement, if within threshold, return number of frame interpolation
    pfm_file_name = f"into_future/{side}/OpticalFlowIntoFuture_{img_name[:-4]}_{letter}.pfm"
    optical_flow_file_path = os.path.join(flow_root, scene_name, pfm_file_name)
    optical_flow = read_pfm(optical_flow_file_path)[0]

    distance = np.sqrt(optical_flow[..., 0]**2+optical_flow[..., 1]**2)
    step_size = np.max(distance)

    # Temporary threshold:
    # if not 170 > step_sz > 100:
    #     return None, None
    if step_size > max_pxl_step:
        return None, None
    max_flow = np.max((step_size, min_nb))/scale_reduct
    if max_flow <= 37:
        return 23, max_flow
    else:
        return 47, max_flow


def load_cam_models(device: torch.device) -> ([camera_model], [Tensor]):
    psf49: Tensor = torch.load('data/preprocess/learned_code/learned_code_27.pt').to(device)
    psf25: Tensor = torch.load('data/preprocess/learned_code/learned_code_S25_27.pt').to(device)
    cam49: camera_model = camera_model(psf49.shape).to(device)
    cam25: camera_model = camera_model(psf25.shape).to(device)
    return [cam25, cam49], [psf25, psf49]


def apply_psf(cam: list, psfs: list, batch: torch.Tensor, nb_imgs: int, do_apply_gamma: bool=True):
    # Apply PSF convolution to the stacked images
    if nb_imgs == 23:
        blurred_image = cam[0](batch, psfs[0])
    else:
        blurred_image = cam[1](batch, psfs[1])

    if do_apply_gamma:
        blurred_image = apply_gamma(blurred_image)

    blurred_image = transforms.ToPILImage()(blurred_image).convert("RGB")

    return blurred_image
