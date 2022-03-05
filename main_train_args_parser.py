import argparse

import data
from FlowNetPytorch import models


def parse_arguments():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__"))
    dataset_names = sorted(name for name in data.__all__)
    parser: argparse.ArgumentParser = \
        argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
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
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                        help='solver algorithms')
    parser.add_argument('--data_path', '-dp', default='',
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
    parser.add_argument('--multiscale-weights', '-w', default=[0.005, 0.01, 0.02, 0.08, 0.32], type=float, nargs=5,
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
                        help='don\'t append date timestamp to folder')
    parser.add_argument('--div_flow', default=20, type=float,
                        help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
    parser.add_argument('--milestones', default=[30, 60, 80], metavar='N', nargs='*',
                        help='epochs at which learning rate is divided by 2')


    args: argparse.Namespace = parser.parse_args()
    return args

