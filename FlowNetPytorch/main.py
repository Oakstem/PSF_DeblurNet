import argparse
import os
import time
import wandb
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import models
from models.Framing import GoWithTheFlownet
from models.raft import RAFT
import datasets
from datasets.load_data import load_data
from multiscaleloss import multiscaleEPE, realEPE
import datetime
from torch.utils.tensorboard import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint, generate_img
from torch.nn import SmoothL1Loss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('data', metavar='DIR',default="../..",
#                     help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument(
    "--split_seed",
    type=int,
    default=None,
    help="Seed the train-val split to enforce reproducibility (consistent restart too)",
)
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownetc',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('--data_path','-dp', default='',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--limit', default=0.9, type=float,
                    help='limit the dataset size')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print_freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default="",
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_false',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div_flow', default=20, type=float,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')



best_EPE = -1
n_iter = int(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_EPE, n_iter
    try:
        import colab
        data_path = "/content/drive/MyDrive/test_chairs"
    except:
        data_path = "G:/My Drive/Colab Notebooks/test_chairs"

    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
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
    # train_writer = SummaryWriter(os.path.join(save_path,'train'))
    # test_writer = SummaryWriter(os.path.join(save_path,'test'))
    output_writers = []
    # for i in range(3):
    #     output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))

    # Data loading code
    # input_transform = transforms.Compose([
    #     flow_transforms.ArrayToTensor(),
    #     transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
    #     transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])
    # ])
    # target_transform = transforms.Compose([
    #     flow_transforms.ArrayToTensor(),
    #     transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    # ])
    #
    # if 'KITTI' in args.dataset:
    #     args.sparse = True
    # if args.sparse:
    #     co_transform = flow_transforms.Compose([
    #         flow_transforms.RandomCrop((320,448)),
    #         flow_transforms.RandomVerticalFlip(),
    #         flow_transforms.RandomHorizontalFlip()
    #     ])
    # else:
    #     co_transform = flow_transforms.Compose([
    #         flow_transforms.RandomTranslate(10),
    #         flow_transforms.RandomRotate(10,5),
    #         flow_transforms.RandomCrop((320,448)),
    #         flow_transforms.RandomVerticalFlip(),
    #         flow_transforms.RandomHorizontalFlip()
    #     ])

    print("=> fetching img pairs in '{}'".format(data_path))
    # train_set, test_set = datasets.__dict__[args.dataset](
    #     args.data,
    #     transform=input_transform,
    #     target_transform=target_transform,
    #     co_transform=co_transform,
    #     split=args.split_file if args.split_file else args.split_value
    # )
    # print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
    #                                                                        len(train_set),
    #                                                                        len(test_set)))
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(
    #     test_set, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True, shuffle=False)
    # data_path = "G:/My Drive/Colab Notebooks/test_chairs/"
    # data_path = os.path.abspath(data_path)

    train_loader = load_data(data_path, args.batch_size, train=True, shuffle=True, limit=args.limit)
    val_loader = load_data(data_path, args.batch_size, train=False, shuffle=True, limit=args.limit)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GoWithTheFlownet(device)
    model.to(device)
    # create model
    if args.pretrained:
        network_data = torch.load(os.path.join("checkpoints", args.pretrained), map_location=device)
        args.arch = network_data['arch']
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

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    if args.evaluate:
        best_EPE = validate(val_loader, model, raft_model, 0, output_writers, experiment)
        return




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
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

            # train_loss, train_EPE, experiment = train(train_loader, model, optimizer, epoch, train_writer, experiment)
        experiment = train(train_loader, model, raft_model, optimizer, epoch, train_writer, experiment)
        # train_writer.add_scalar('mean EPE', train_EPE, epoch)

        scheduler.step()
        # evaluate on validation set
        with torch.no_grad():
            experiment, EPE = validate(val_loader, model, raft_model, epoch, output_writers, experiment)


        # test_writer.add_scalar('mean EPE', EPE, epoch)

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
    with tqdm(total=len(train_loader)*args.batch_size, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
        for i, (input, target, idx) in enumerate(train_loader):
            # measure data loading time
            # data_time.update(time.time() - end)
            target = target.to(device)
            # input = torch.cat(input,1).to(device)
            input = input.to(device)

            # compute output
            frame1, frame2 = model(input)
            flow = flow_model(frame1, frame2, iters=20, test_mode=True)
            if args.sparse:
                # Since Target pooling is not very precise when sparse,
                # take the highest resolution prediction and upsample it instead of downsampling target
                h, w = target.size()[-2:]
                output = [F.interpolate(output[0], (h,w)), *output[1:]]

            # loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
            loss = criterion(flow[1], target)
            # flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
            # record loss and EPE
            # losses.update(loss.item(), target.size(0))
            # train_writer.add_scalar('train_loss', loss.item(), n_iter)
            # flow2_EPEs.update(flow2_EPE.item(), target.size(0))

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()


            pbar.update(input.shape[0])
            experiment.log({
                'train loss': loss.item(),
                'step': n_iter,
                'epoch': epoch
            })
            pbar.set_postfix(**{'loss (batch)': loss.item()})
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
    flow2_EPE = 0
    criterion = SmoothL1Loss()
    # end = time.time()
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.to(device)
        # input = torch.cat(input,1).to(device)
        input = input.to(device)

        # compute output
        # output = model(input)
        frame1, frame2 = model(input)
        flow = flow_model(frame1, frame2, iters=20, test_mode=True)

        # flow2_EPE += args.div_flow*realEPE(output, target, sparse=args.sparse)
        flow2_EPE += args.div_flow*criterion(flow[1], target)
        # record EPE
        # flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        max_val = 20
        # if i < len(output_writers):  # log first output of first batches
        # if args.evaluate:
        #     generate_img(target, flow, max_val, args.div_flow)

        # if i < 1:
    avg_epe = (flow2_EPE/len(val_loader)).item()
    logging.info('Validation epe score: {}'.format(avg_epe))
    experiment.log({
        'learning rate': args.lr,
        'validation loss': avg_epe,
        'images': wandb.Image(input[0].cpu()),
        # 'flows': {
        #     'true': wandb.Image(flow2rgb(args.div_flow * target[0], max_value=max_val).transpose((1,2,0))),
        #     'pred': wandb.Image(flow2rgb(args.div_flow * output[0], max_value=max_val).transpose((1,2,0))),
        # },
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