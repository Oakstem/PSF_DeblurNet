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

from main_train_flownet_args_parser import parse_arguments
from data.load_data import load_data
from FlowNetPytorch.models.Framing import GoWithTheFlownet
from FlowNetPytorch.models.raft import RAFT
import datetime
from FlowNetPytorch.util import flow2rgb, save_checkpoint
from torch.nn import SmoothL1Loss


best_EPE = -1
n_iter = int(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global best_EPE, n_iter
    try:
        import colab
        data_path = "/content/drive/MyDrive/test_chairs"
    except:
        data_path = "G:/My Drive/Colab Notebooks/test_chairs"
        data_path = "/home/jupyter/"

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GoWithTheFlownet(device)
    model.to(device)

    # create model
    if args.pretrained:
        network_data = torch.load(os.path.join("checkpoints", args.pretrained), map_location=device)
        args.arch = network_data['arch']
        model.load_state_dict(network_data['state_dict'])
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    # model = models.__dict__[args.arch](network_data).to(device)

    args.small = True
    args.mixed_precision = False
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load("models/raft-small.pth", map_location=device))

    raft_model = raft_model.module
    raft_model.to(device)
    raft_model.eval()

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    # param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
    #                 {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]
    param_groups = [{'params': model.parameters(), 'weight_decay': args.weight_decay}]

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
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
                Device:          {device.type}
                Images scaling:  {1}
                Mixed Precision: {False}
            ''')
    args.div_flow = 20
    args.nb_raft_iter = 8
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

            # train_loss, train_EPE, experiment = train(train_loader, model, optimizer, epoch, train_writer, experiment)
        if not args.evaluate:
            experiment = train(train_loader, model, raft_model, optimizer, epoch, train_writer, experiment)
        # train_writer.add_scalar('mean EPE', train_EPE, epoch)

            scheduler.step()
        # evaluate on validation set
        with torch.no_grad():
            experiment, EPE = validate(val_loader, model, raft_model, epoch, output_writers, experiment)

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
            'state_dict': model.module.state_dict(),
            # 'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best=is_best, save_path=save_path)



def train(train_loader, model, flow_model, optimizer, epoch, train_writer, experiment):
    global n_iter, args
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model = model.module
    model.train()
    model = model.to(device)
    flow_model.eval()
    flow_model = flow_model.to(device)

    criterion = SmoothL1Loss()
    end = time.time()
    loss_weights = [0.20, 0.35, 0.45]
    with tqdm(total=len(train_loader)*args.batch_size, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:

        tot_loss = 0
        for i, (input, target, idx) in enumerate(train_loader):
            # measure data loading time
            # data_time.update(time.time() - end)
            target = target.to(device)
            # input = torch.cat(input,1).to(device)
            input = input.to(device)

            # compute output
            frame1, frame2 = model(input)
            flow = flow_model(frame1[0], frame2[0], iters=args.nb_raft_iter, test_mode=True)
            sm_tgt = tr_f.resize(target, flow[0].shape[-1])
            loss1 = loss_weights[-1] * args.div_flow * criterion(flow[0], sm_tgt)
            loss2 = loss_weights[0] * args.div_flow * criterion(flow[1], target)
            loss = loss1 + loss2
            tot_loss += loss.item()
            # print(f"Actual Loss:{loss}")
            # for ind in range(len(frame1)):
            #     sz = frame1[ind].shape[-1]//8
            #     if sz > 4:
            #         print(f"index:{ind}")
            #         tgt = tr_f.center_crop(target, sz * 8)
            #         flow = flow_model(frame1[ind], frame2[ind], iters=8, test_mode=True)
            #
            #         loss = loss_weights[ind]*args.div_flow*criterion(flow[1], tgt)
            #
            #         loss.backward(retain_graph=True)
            #

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)

            # flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
            # record loss and EPE
            # losses.update(loss.item(), target.size(0))
            # train_writer.add_scalar('train_loss', loss.item(), n_iter)
            # flow2_EPEs.update(flow2_EPE.item(), target.size(0))

            # compute gradient and do optimization step

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()


            pbar.update(input.shape[0])
            experiment.log({
                'train loss': tot_loss/(i+1),
                'step': n_iter,
                'epoch': epoch
            })
            pbar.set_postfix(**{'loss (batch)': tot_loss/(i+1)})
            # if i % args.print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
            #           .format(epoch, i, epoch_size, batch_time,
            #                   data_time, losses, flow2_EPEs))
            # if i >= epoch_size:
            #     break
            n_iter += 1
    # return losses.avg, flow2_EPEs.avg, experiment
    return experiment


def validate(val_loader, model, flow_model, epoch, output_writers, experiment):
    global args

    # batch_time = AverageMeter()
    # flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    flow_model.eval()
    model.to(device)
    flow_model.to(device)
    criterion = SmoothL1Loss()
    loss_weights = [0.20, 0.35, 0.45]
    tot_loss = 0
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.to(device)
        # input = torch.cat(input,1).to(device)
        input = input.to(device)

        # compute output
        frame1, frame2 = model(input)
        flow = flow_model(frame1[0], frame2[0], iters=args.nb_raft_iter, test_mode=True)
        sm_tgt = tr_f.resize(target, flow[0].shape[-1])
        loss1 = loss_weights[-1] * args.div_flow * criterion(flow[0], sm_tgt)
        loss2 = loss_weights[0] * args.div_flow * criterion(flow[1], target)
        loss = loss1 + loss2
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
            # if epoch == args.start_epoch:
            #     # mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
            #     mean_values = torch.tensor([0., 0., 0.], dtype=input.dtype).view(3, 1, 1)
            #     output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=max_val), 0)
            #     output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
            #     # output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
            # output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=max_val), epoch)

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
        #           .format(i, len(val_loader), batch_time, flow2_EPEs))

    # print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    # return flow2_EPEs.avg, experiment
    return experiment, avg_epe


if __name__ == '__main__':
    main()