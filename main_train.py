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
from torch.nn import SmoothL1Loss


best_EPE = -1
n_iter = int(0)


def main():
    global best_EPE, n_iter
    try:
        import colab
        data_path = "/content/drive/MyDrive/test_chairs"
    except:
        data_path = "G:/My Drive/Colab Notebooks/test_chairs"
        # data_path = "/home/jupyter/"

    args: argparse.Namespace = parse_arguments()

    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch, args.solver, args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size, args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.split_seed is not None:
        np.random.seed(args.split_seed)

    # (Initialize logging)
    experiment = wandb.init(project='FlowNet', resume='allow', anonymous='must')        #, entity="oakstem")

    experiment.config.update(dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                                  val_percent=0.1, save_checkpoint=True, img_scale=1,
                                  amp=False))
    train_writer = []
    test_writer = []

    output_writers = []


    print("=> fetching img pairs in '{}'".format(data_path))

    train_loader = load_data(data_path, args.batch_size, train=True, shuffle=True, limit=args.limit)
    val_loader = load_data(data_path, args.batch_size, train=False, shuffle=True, limit=args.limit)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_flownet = GoWithTheFlownet(args.device)
    model_flownet.to(args.device)

    # create model
    if args.pretrained:
        network_data = torch.load(os.path.join("FlowNetPytorch", "checkpoints", args.pretrained), map_location=args.device)
        args.arch = network_data['arch']
        model_flownet.load_state_dict(network_data['state_dict'])
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    # model = models.__dict__[args.arch](network_data).to(args.device)

    args.small = False
    args.mixed_precision = False
    model_raft = torch.nn.DataParallel(RAFT(args))
    model_raft.load_state_dict(torch.load(
        os.path.join("FlowNetPytorch", "models", "raft-sintel.pth"), map_location=args.device))

    model_raft = model_raft.module
    model_raft.to(args.device)
    model_raft.eval()

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    # param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
    #                 {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]
    param_groups = [{'params': model_flownet.parameters(), 'weight_decay': args.weight_decay}]

    if args.device.type == "cuda":
        model_flownet = torch.nn.DataParallel(model_flownet).cuda()
        cudnn.benchmark = True

    optimizer: torch.optim.Optimizer = None
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    # if args.evaluate:
    #     best_EPE = validate(val_loader, model, raft_model, 0, output_writers, experiment)
    #     return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

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
                Mixed Precision: {False}
            ''')
    args.div_flow = 20
    args.nb_raft_iter = 8
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

            # train_loss, train_EPE, experiment = train(train_loader, model, optimizer, epoch, train_writer, experiment)
        if not args.evaluate:
            experiment = train(args, train_loader, model_flownet, model_raft, optimizer, epoch, train_writer, experiment)
        # train_writer.add_scalar('mean EPE', train_EPE, epoch)

            scheduler.step()
        # evaluate on validation set
        with torch.no_grad():
            experiment, EPE = validate(args, val_loader, model_flownet, model_raft, epoch, output_writers, experiment)

        if best_EPE < 0:
            best_EPE = EPE
        #
        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        if is_best:
            print("Best eval EPE recorded!")
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_flownet.module.state_dict(),
            # 'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best=is_best, save_path=save_path)



def train(args, train_loader, model_flownet, model_raft, optimizer, epoch, train_writer, experiment):
    global n_iter
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    try:
        model_flownet = model_flownet.module
    except:
        None
    model_flownet.train()
    model_flownet = model_flownet.to(args.device)

    model_raft.eval()
    model_raft = model_raft.to(args.device)

    criterion = SmoothL1Loss()
    end = time.time()
    loss_weights = [0.04, 0.08, 0.1, 0.13, 0.15, 0.2, 0.3]
    flow_scales = [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    with tqdm(total=len(train_loader)*args.batch_size, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:

        tot_loss = 0
        for i, (input, target, idx) in enumerate(train_loader):
            # measure data loading time
            # data_time.update(time.time() - end)
            target = target.to(args.device)
            # input = torch.cat(input,1).to(args.device)
            input = input.to(args.device)

            # compute output
            frame1, frame2 = model_flownet(input)
            flow1 = model_raft(frame1[0], frame2[0], iters=args.nb_raft_iter, test_mode=True)
            flow2 = model_raft(frame1[1], frame2[1], iters=args.nb_raft_iter, test_mode=True)
            flow3 = model_raft(frame1[2], frame2[2], iters=args.nb_raft_iter, test_mode=True)
            flow4 = model_raft(frame1[3], frame2[3], iters=args.nb_raft_iter, test_mode=True)
            flow5 = model_raft(frame1[4], frame2[4], iters=args.nb_raft_iter, test_mode=True)
            flow6 = model_raft(frame1[5], frame2[5], iters=args.nb_raft_iter, test_mode=True)

            # sm_tgt = tr_f.resize(target, flow1[0].shape[-1])
            # sm_tgt2 = tr_f.resize(target, flow2[0].shape[-1])
            # sm_tgt3 = tr_f.resize(target, flow3[0].shape[-1])
            target_128 = tr_f.resize(target, flow3[1].shape[-1])

            # loss1_1 = loss_weights[0] * args.div_flow * criterion(flow[0], sm_tgt)
            loss1_1 = loss_weights[0] * args.div_flow * criterion(flow1[1], flow_scales[0]*target)

            # loss1_2 = loss_weights[1] * args.div_flow * criterion(flow2[0], sm_tgt2)
            loss2_1 = loss_weights[1] * args.div_flow * criterion(flow2[1], flow_scales[1]*target)

            # loss1_3 = loss_weights[2] * args.div_flow * criterion(flow3[0], sm_tgt3)
            loss3_1 = loss_weights[2] * args.div_flow * criterion(flow3[1], flow_scales[2] * target_128)
            loss4_1 = loss_weights[3] * args.div_flow * criterion(flow4[1], flow_scales[3] * target_128)
            loss5_1 = loss_weights[4] * args.div_flow * criterion(flow5[1], flow_scales[4] * target_128)
            loss6_1 = loss_weights[5] * args.div_flow * criterion(flow6[1], flow_scales[5] * target_128)

            loss = loss1_1 + loss2_1 + loss3_1 + loss4_1 + loss5_1 + loss6_1
            tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(input.shape[0])
            experiment.log({
                'train loss': tot_loss/(i+1),
                'step': n_iter,
                'epoch': epoch
            })
            pbar.set_postfix(**{'loss (batch)': tot_loss/(i+1)})

            n_iter += 1
    return experiment


def validate(args, val_loader, model_flownet, model_raft, epoch, output_writers, experiment):

    # switch to evaluate mode
    model_flownet.eval()
    model_raft.eval()

    model_flownet.to(args.device)
    model_raft.to(args.device)

    criterion = SmoothL1Loss()
    loss_weights = [0.20, 0.35, 0.45]
    tot_loss = 0
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.to(args.device)
        # input = torch.cat(input,1).to(args.device)
        input = input.to(args.device)

        # compute output
        frame1, frame2 = model_flownet(input)
        flow = model_raft(frame1[0], frame2[0], iters=args.nb_raft_iter, test_mode=True)
        # sm_tgt = tr_f.resize(target, flow[0].shape[-1])
        # loss1 = args.div_flow * criterion(flow[0], sm_tgt)
        loss2 = args.div_flow * criterion(flow[1], target)
        loss = loss2 #+ loss1
        tot_loss += loss.item()
        # print(f"flow[1] shape:{flow[1].shape}")
        # flow2_EPE += args.div_flow*realEPE(output, target, sparse=args.sparse)
        # flow2_EPE += args.div_flow*criterion(flow[1], target)
        # record EPE
        # flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i < len(output_writers):  # log first output of first batches
        # if args.evaluate:
        #     generate_img(target, flow, max_val, args.div_flow)

        # if i < 1:
    max_val = 10
    avg_epe = (tot_loss/len(val_loader))
    logging.info('Validation epe score: {}'.format(avg_epe))
    experiment.log({
        'learning rate': args.lr,
        'validation loss': avg_epe,
        'images': wandb.Image(input[0].cpu()),
        'flows': {
            'true': wandb.Image(flow2rgb(args.div_flow * target[0], max_value=max_val).transpose((1,2,0))),
            'pred': wandb.Image(flow2rgb(args.div_flow * flow[1][0], max_value=max_val).transpose((1,2,0))),
        },
        'step': n_iter,
        'epoch': epoch,
    })
    return experiment, avg_epe


if __name__ == '__main__':
    main()
