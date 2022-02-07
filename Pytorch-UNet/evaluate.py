import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from utils.dice_score import DiceLoss
from utils.utils import compute_loss
from utils.losses import MultiScale


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    tot_loss = 0
    ce_loss = CrossEntropyLoss()
    nb_classes = net.n_classes
    dice_loss = DiceLoss(nb_classes // 2)
    mult_loss = MultiScale()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # mask_true = F.one_hot(mask_true, nb_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if nb_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # mask_pred = F.one_hot(mask_pred.argmax(dim=1), nb_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                # loss = compute_loss(mask_pred, mask_true, ce_loss, dice_loss, nb_classes)
                loss, epe = mult_loss(mask_pred, mask_true)
                tot_loss += loss
           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return tot_loss
    return tot_loss / num_val_batches
