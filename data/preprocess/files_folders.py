import os
from pathlib import Path

from data.type import Type


def get_dataset_path(type: Type, data_path: str, train: bool):
    if type == Type.MONKAA:
        data_path = os.path.join(data_path, "Monkaa")
        target_root = os.path.join(data_path, 'blurred_test')
        flow_root = os.path.join(data_path, 'optical_flow')
        rgb_root = os.path.join(data_path, 'frames_cleanpass')
    elif type == Type.FLYING_CHAIRS2:
        data_path = os.path.join(data_path, "FlyingChairs2")
        target_root = os.path.join(data_path, 'train' if train else 'val')
        flow_root = os.path.join(data_path, 'train' if train else 'val')
        rgb_root = os.path.join(data_path, 'train' if train else 'val')
    else:
        target_root = ""
        flow_root = ""
        rgb_root = ""
        print("Error! Unknown dataset type")

    if target_root != "" and not os.path.exists(target_root):
        Path(target_root).mkdir(parents=True, exist_ok=True)

    return target_root, flow_root, rgb_root


def get_blurred_image_path(target_root: str, scene_name: str, filename: str, side: str):
    image_name: str = get_image_name(filename)
    blurr_path = os.path.join(target_root, f"{scene_name}/{side}/{image_name}")
    return blurr_path, image_name

def get_image_name(filename: str):
    if len(filename.split("/")) > len(filename.split("\\")):
        image_name = filename.split("/")[-1]
    else:
        image_name = filename.split("\\")[-1]

    return image_name

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

def get_filenames_from_subfolders(root: str, side: str):
    images_list = []
    total_images_num = 0

    if side == "right":
        opp_side = "left"
    else:
        opp_side = "right"

    for sub, dirs, files in os.walk(root):
        # Discard the "right" dirs since no stereo is needed
        if not dirs and opp_side not in sub:
            file_list = [os.path.join(sub, f) for f in files]
            file_list.sort()
            images_list += [file_list]
            total_images_num += len(file_list)

    print(f"Total number of images to process:{total_images_num}")
    return images_list, total_images_num


def get_filenames_by_extention(dir, files_extention: str):
        files = []
        for f in os.scandir(dir):
            if f.is_file():
                if files_extention in f.name:
                    files.append(f.path)

        return files
