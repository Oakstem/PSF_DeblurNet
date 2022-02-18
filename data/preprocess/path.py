import os
from pathlib import Path


def get_dataset_path(data_type: str, root: str):
    if "monkaa" in data_type.lower():
        target_root = os.path.join(root, 'Monkaa/blurred_test')
        flow_root = os.path.join(root, 'Monkaa/optical_flow')
        rgb_root = os.path.join(root, 'Monkaa/frames_cleanpass')
        Path(target_root).mkdir(parents=True, exist_ok=True)
    elif "flying_chairs2" in data_type.lower():
        target_root = os.path.join(root, 'FlyingChairs2/train')
        flow_root = os.path.join(root, 'FlyingChairs2/train')
        rgb_root = os.path.join(root, 'FlyingChairs2/train')
        Path(target_root).mkdir(parents=True, exist_ok=True)
    else:
        target_root = ""
        flow_root = ""
        rgb_root = ""
        print("Error! Unknown dataset type")

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
