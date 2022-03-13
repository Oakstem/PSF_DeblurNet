import argparse
import os
import time

import numpy as np
import wandb
import logging
from tqdm import tqdm
import torch
import torchvision.transforms.functional as tr_f
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from main_train_args_parser import parse_arguments
from data.load_data import load_data
from FlowNetPytorch.models.Framing import GoWithTheFlownet
from FlowNetPytorch.models.raft import RAFT
import datetime
from FlowNetPytorch.util import flow2rgb, save_checkpoint
from FlowNetPytorch.models.util import warp_loss
from torch.nn import SmoothL1Loss

# from PWC_Net.PyTorch import models

best_EPE = -1
n_iter = int(0)


def main():
    global best_EPE, n_iter
    args: argparse.Namespace = parse_arguments()

    try:
        import colab
        # data_path = "/content/drive/MyDrive/test_chairs"
        if args.dataset == 'monkaa':
            data_path = "/content/drive/MyDrive/test_chairs"
        else:
            data_path = "/content/drive/MyDrive/Colab Notebooks"
    except:
        data_path = "/home/gentex/studies"

    save_path = '{},{},{}epochs{},b{},lr{},lim{}'.format(
        args.arch, args.solver, args.epochs,
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size, args.lr, args.limit)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset, save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.split_seed is not None:
        np.random.seed(args.split_seed)

    # (Initialize logging)
    wandb_log = wandb.init(project='GoWithTheFlowNet', resume='allow', anonymous='must')

    wandb_log.config.update(dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                                 val_percent=0.1, save_checkpoint=True, img_scale=1, amp=False))

    print("=> fetching img pairs in '{}'".format(data_path))

    train_loader = load_data(data_path, args.batch_size, train=True, shuffle=True, limit=args.limit,
                             data_type=args.dataset)
    val_loader = load_data(data_path, args.batch_size, train=False, shuffle=True, limit=args.limit,
                           data_type=args.dataset)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_flownet = GoWithTheFlownet(args.device)
    model_flownet.to(args.device)

    # create model
    if args.pretrained:
        network_data = torch.load(os.path.join("FlowNetPytorch", "checkpoints", args.pretrained),
                                  map_location=args.device)
        args.arch = network_data['arch']
        model_flownet.load_state_dict(network_data['state_dict'])
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    # model = models.__dict__[args.arch](network_data).to(args.device)

    if args.estm_net == 'raft':
        args.small = True
        args.mixed_precision = False
        model_raft = torch.nn.DataParallel(RAFT(args))
        if args.small:
            raft_path = "raft-small.pth"
        else:
            raft_path = "raft-sintel.pth"
        model_raft.load_state_dict(torch.load(
            os.path.join("FlowNetPytorch", "models", raft_path), map_location=args.device))

        model_raft = model_raft.module
        model_raft.to(args.device)
        model_raft.eval()
        flow_estimator = model_raft
    else:
        pwc_model_fn = '/home/gentex/studies/PSF_DeblurNet/PWC_Net/PyTorch/pwc_net.pth.tar'
        flow_estimator = models.pwc_dc_net(pwc_model_fn)
        flow_estimator = flow_estimator.cuda()
    flow_estimator.eval()

    assert (args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    # param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
    #                 {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]
    param_groups = [{'params': model_flownet.parameters(), 'weight_decay': args.weight_decay}]

    if args.device.type == "cuda":
        model_flownet = torch.nn.DataParallel(model_flownet).cuda()
        cudnn.benchmark = True

    optimizer: torch.optim.Optimizer = None
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0, verbose=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Starting training:
                Epochs:          {args.epochs}
                Batch size:      {args.batch_size}
                Learning rate:   {args.lr}
                Training size:   {len(train_loader) * args.batch_size}
                Validation size: {len(val_loader) * args.batch_size}
                Checkpoints:     {True}
                Device:          {args.device.type}
                Images scaling:  {1}
                Mixed Precision: {False}''')
    args.div_flow = 20
    args.nb_raft_iter = 8
    args.upscale = None
    args.unsupervised = False

    for epoch in range(args.start_epoch, args.epochs):
        if not args.evaluate:
            train_one_epoch(args, train_loader, model_flownet, flow_estimator, optimizer, epoch, wandb_log)
            scheduler.step()
            print(scheduler.get_last_lr())
            wandb_log.log({'Learning rate': scheduler.get_last_lr()})

        # evaluate on validation set
        with torch.no_grad():
            EPE = validate(args, val_loader, model_flownet, flow_estimator, epoch, wandb_log)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        if is_best:
            print("Best eval EPE recorded!")
        save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model_flownet.module.state_dict(),
                         'learning rate': scheduler.get_last_lr(), 'div_flow': args.div_flow},
                        is_best=is_best, save_path=save_path)


def train_one_epoch(args, train_loader, model_flownet, model_raft, optimizer, epoch, wandb_log):
    global n_iter

    try:
        model_flownet = model_flownet.module
    except:
        pass

    # switch to train mode
    model_flownet.train()
    model_flownet = model_flownet.to(args.device)

    if args.estm_net == "raft":
        model_raft = model_raft.to(args.device)
    model_raft.eval()

    criterion = SmoothL1Loss()
    loss_weights = [0.22, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]
    flow_scales = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    with tqdm(total=len(train_loader) * args.batch_size, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:

        tot_loss = 0
        for i, (input, target, idx) in enumerate(train_loader):

            target = target.to(args.device)
            input = input.to(args.device)
            if args.upscale:
                input = tr_f.resize(input, args.upscale)

            # compute output
            frame1, frame2, feat1, feat2 = model_flownet(input)

            if args.estm_net == 'raft':
                flow1 = model_raft(frame1, frame2)
            else:
                cat_frames = torch.cat((frame1, frame2), dim=1)
                flow1 = model_raft(cat_frames)

            if not args.unsupervised:
                loss = 0
                for i in range(12):
                    loss += loss_weights[i] * criterion(flow1[i], flow_scales[0] * target)

            tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(input.shape[0])
            wandb_log.log({'train loss': tot_loss / (i + 1), 'step': n_iter, 'epoch': epoch})
            pbar.set_postfix(**{'loss (batch)': tot_loss / (i + 1)})

            n_iter += 1


def validate(args, val_loader, model_flownet, model_raft, epoch, wandb_log):
    # switch to evaluate mode
    model_flownet.eval()
    model_raft.eval()

    model_flownet.to(args.device)
    model_raft.to(args.device)

    criterion = SmoothL1Loss()
    tot_loss = 0
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.to(args.device)
        # input = torch.cat(input,1).to(args.device)
        input = input.to(args.device)
        if args.upscale:
            input = tr_f.resize(input, args.upscale)
        # compute output
        frame1, frame2, feat1, feat2 = model_flownet(input)

        if args.estm_net == 'raft':
            flow1 = model_raft(frame1, frame2)
            flow1 = flow1[5]
        else:
            cat_frames = torch.cat((frame1, frame2), dim=1)
            flow1 = model_raft(cat_frames)

        flows = flow1
        if not args.unsupervised:
            # target_64 = tr_f.resize(target, flow1[0].shape[-1])
            loss = criterion(args.div_flow * flow1, args.div_flow * target)
            if loss < 2:
                stop = 1
        else:
            loss = warp_loss(feat1, feat2, flows, criterion)
        tot_loss += loss.item()

    max_val = 10
    avg_epe = (tot_loss / len(val_loader))
    logging.info('Validation epe score: {}'.format(avg_epe))
    wandb_log.log({'learning rate': args.lr,
                   'validation loss': avg_epe,
                   'images': wandb.Image(input[0].cpu()),
                   'flows': { 'true': wandb.Image(flow2rgb(args.div_flow * target[0],
                                                           max_value=max_val).transpose((1, 2, 0))),
                              'pred': wandb.Image(flow2rgb(2 * args.div_flow * flow1[0],
                                                           max_value=max_val).transpose((1, 2, 0))),},
                   'step': n_iter,
                   'epoch': epoch,})

    return avg_epe


def show_results(target, pred, div_flow):
    import matplotlib.pyplot as plt
    target = flow2rgb(div_flow * target, 10).transpose((1, 2, 0))
    pred = flow2rgb(div_flow * pred, 10).transpose((1, 2, 0))
    conc = np.hstack((target, pred))
    plt.figure()
    plt.imshow(conc)
    done = 1


if __name__ == '__main__':
    main()
