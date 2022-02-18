import os
from pathlib import Path

from data.type import Type


def get_dataset_path(type: Type, data_path: str):
    if type == Type.MONKAA:
        data_path = os.path.join(data_path, "Monkaa")
        target_root = os.path.join(data_path, 'blurred_test')
        flow_root = os.path.join(data_path, 'optical_flow')
        rgb_root = os.path.join(data_path, 'frames_cleanpass')
    elif type == Type.FLYING_CHAIRS2:
        data_path = os.path.join(data_path, "FlyingChairs2")
        target_root = os.path.join(data_path, 'train')
        flow_root = os.path.join(data_path, 'train')
        rgb_root = os.path.join(data_path, 'train')
    else:
        target_root = ""
        flow_root = ""
        rgb_root = ""
        print("Error! Unknown dataset type")

    if target_root != "" and not os.path.exists(target_root):
        Path(target_root).mkdir(parents=True, exist_ok=True)

    return target_root, flow_root, rgb_root


def get_blurred_image_path(target_root: str, scene_name: str, filename: str, side: str):
    if len(filename.split("/")) > len(filename.split("\\")):
        img_name = filename.split("/")[-1]
    else:
        img_name = filename.split("\\")[-1]
    blr_path = os.path.join(target_root, f"{scene_name}/{side}/{img_name}")
    return blr_path, img_name


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
