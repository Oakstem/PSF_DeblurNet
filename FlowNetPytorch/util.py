import os
import numpy as np
import shutil
import torch
import matplotlib.pyplot as plt


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    # if flow_map.max() > max_value:
    #     print(f"Error! actually the max value is: {flow_map.max()}")
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def generate_img(flow1, flow2, max_val, div_flow):
    # pred_flow_img = flow2color(flow1.squeeze().permute((1, 2, 0)))
    # gt_flow_img = flow2color(flow2.squeeze().permute((1, 2, 0)))
    fig, axs = plt.subplots(1, 2)
    for idx in range(len(flow1)):
        pred_flow_img = flow2rgb(div_flow * flow2[idx], max_val).transpose((1, 2, 0))
        gt_flow_img = flow2rgb(div_flow * flow1[idx], max_val).transpose((1, 2, 0))
        axs[0].imshow(gt_flow_img)
        axs[1].imshow(pred_flow_img)
        axs[0].set_title("Ground Truth")
        axs[1].set_title("Predicted")