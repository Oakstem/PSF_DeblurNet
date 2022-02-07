import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def compute_loss(outputs, labels, ce_loss, dice_loss, nb_classes: int):
    u, v = outputs[:, :nb_classes // 2, :, :], outputs[:, nb_classes // 2:, :, :]
    # Currently clipping all OF values to be 0 to 99
    loss_ce1 = ce_loss(u, labels[:, 0, :, :].long().clip(min=0, max=nb_classes // 2 - 1))
    loss_ce2 = ce_loss(v, labels[:, 1, :, :].long().clip(min=0, max=nb_classes // 2 - 1))
    loss_dice1 = dice_loss(u, labels[:, 0, :, :], softmax=True)
    loss_dice2 = dice_loss(v, labels[:, 1, :, :], softmax=True)
    loss = 0.4 * (loss_ce1 + loss_ce2) + 0.6 * (loss_dice1 + loss_dice2)

    return loss
