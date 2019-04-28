import argparse
import collections
import datetime
import imp
import os
import pickle
import time
import lmdb
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader.dataset import ImagnetVIDDataset
from network.SiamRPN import *
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger
from utils.loss import rpn_cross_entropy_balance, rpn_smoothL1

################################################
# This python file contains four parts:
#
# Part 1. Argument Parser
# Part 2. configurations:
#                       Part 2-1. Basic configuration
#                       Part 2-2. dataloader instantiation
#                       Part 2-3. log configuration
#                       Part 2-4. configurations for loss function, SiamRPN++ network, and optimizer
# Part 3. 'train' function
# Part 4. 'main' function
################################################


# Part 1. Argument Parser
parser = argparse.ArgumentParser(description='train_SiamRPN++')
parser.add_argument("--exp", type=str, default="test", help="experiment")
parser.add_argument("--data_dir", type=str,
                    default="./VID_2015_RPN++", help="lmdb_dir")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument('--gpu', type=str, default="0,1,2,3", help='choose GPU')
#
parser.add_argument("--training_pairs_with_annotations",
                    type=str, default="", help="training_pairs_with_annotations")
parser.add_argument("--random_shift_factor", type=int,
                    default=32, help="random_shift_factor")
parser.add_argument("--examplar_img_size", type=int,
                    default=127, help="examplar_img_size")
parser.add_argument("--search_region_img_size", type=int,
                    default=255, help="search_region_img_size")
parser.add_argument("--response_map_size", type=int,
                    default=25, help="response_map_size")

args = parser.parse_args()


# Part 2. configurations

# Part 2-1. Basic configuration
basic_configs = collections.OrderedDict()
basic_configs['serial_number'] = args.exp
basic_configs['learning_rate'] = 1e-3
basic_configs['num_epochs'] = 10000
basic_configs["lr_protocol_4_warm_up"] = [(5, 1e-3)]
basic_configs["lr_protocol_after_warm_up"] = [
    (epo, 0.005 * 0.85 ** (epo - 6)) for epo in range(6, 21)]
basic_configs["lr_protocol"] = basic_configs["lr_protocol_4_warm_up"] + \
    basic_configs["lr_protocol_after_warm_up"]
basic_configs["display_step"] = 10
lr_protocol = basic_configs["lr_protocol"]


# Part 2-2. dataloader instantiation
dataloader_configs = collections.OrderedDict()
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers
print '==> training pair url loading......'
#load_file = open(args.training_pairs_with_annotations, 'rb')
#dataloader_configs['training_pairs_with_annotations'] = pickle.load(load_file)
print '==> loading finished......'
# TODO
dataloader_configs['transform'] = transforms.Compose([transforms.ToTensor()])
dataloader_configs['random_shift_factor'] = args.random_shift_factor
dataloader_configs['examplar_img_size'] = args.examplar_img_size
dataloader_configs['search_region_img_size'] = args.search_region_img_size
dataloader_configs['response_map_size'] = args.response_map_size

'''
train_set = TrainingLoader(dataloader_configs)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=dataloader_configs['batch_size'], shuffle=True, num_workers=dataloader_configs['num_workers'])
'''
# TODO
# open lmdb
#ipdb.set_trace()
data_dir = args.data_dir
db = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(200e9))

# create dataset
# -----------------------------------------------------------------------------------------------------
meta_data_path = os.path.join(data_dir, "meta_data.pkl")
meta_data = pickle.load(open(meta_data_path, 'rb'))
all_videos = [x[0] for x in meta_data]
#data_dir = "." + data_dir[15:]
train_dataset = ImagnetVIDDataset(
    db, all_videos, data_dir, dataloader_configs['transform'], dataloader_configs['transform'])
train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True,
                          pin_memory=True, num_workers=dataloader_configs['num_workers'], drop_last=True)

# Part 2-3. log configuration
exp_dir = os.path.join('./experimental_results', args.exp)

exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)

exp_visual_dir = os.path.join(exp_dir, "visual")
if not os.path.exists(exp_visual_dir):
    os.makedirs(exp_visual_dir)

exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
if not os.path.exists(exp_ckpt_dir):
    os.makedirs(exp_ckpt_dir)

now_str = datetime.datetime.now().__str__().replace(' ', '_')
writer_path = os.path.join(exp_visual_dir, now_str)
writer = SummaryWriter(writer_path)

logger_path = os.path.join(exp_log_dir, now_str + ".log")
logger = Logger(logger_path).get_logger()

# TODO
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger.info("basic configuration settings: {}".format(basic_configs))
#logger.info("dataloader configuration settings: {}".format(dataloader_configs))


# Part 2-4. configurations for loss function, SiamRPN++ network, and optimizer

# TODO
loss_function = nn.CrossEntropyLoss()
training_loss = AverageMeter()

net = SiamRPN()
#net = net.cuda()
net = torch.nn.DataParallel(
    net, device_ids=[int(x) for x in args.gpu.split(',')]).cuda()

optimizer = torch.optim.SGD(
    net.parameters(), lr=basic_configs['learning_rate'], momentum=0.9, weight_decay=5e-4)


# Part 3. 'train' function
def train_function(epoch):
    training_loss.reset()
    net.train()

    lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch >
               epoch), lr_protocol[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info("set learning rate to: {}".format(lr))

    for idx, (examplar_img, search_region_img, regression_target, conf_target) in enumerate(train_loader):

        #
        # _BATCH_SIZE = len(examplar_img)   delete

        
        examplar_img = examplar_img.cuda()
        search_region_img = search_region_img.cuda()
        regression_target = regression_target.cuda()
        conf_target = conf_target.cuda()
        optimizer.zero_grad()

        # TODO
        # fused_cls_prediction, fused_regression_prediction = net(examplar_img, search_region_img, _BATCH_SIZE)
        fused_cls_prediction, fused_regression_prediction = net(
            examplar_img, search_region_img)
        # ipdb.set_trace()

        pred_conf = fused_cls_prediction.reshape(
            -1, 2, 5 * 25 * 25).permute(0, 2, 1)
        pred_offset = fused_regression_prediction.reshape(
            -1, 4, 5 * 25 * 25).permute(0, 2, 1)
        # ipdb.set_trace()
        cls_loss = rpn_cross_entropy_balance(
            pred_conf, conf_target, 16, 48, ohem_pos=False, ohem_neg=False)
        reg_loss = rpn_smoothL1(
            pred_offset, regression_target, conf_target, 16, ohem=False)
        batch_loss = cls_loss + 5 * reg_loss

        batch_loss.backward()

        optimizer.step()

        training_loss.update(batch_loss.item())

        if (idx + 1) % basic_configs["display_step"] == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch + 1, idx + 1, len(train_loader)))
            logger.info("current batch loss: {}".format(
                batch_loss.item()))
            logger.info("average loss: {}".format(
                training_loss.avg))

    writer.add_scalars("loss", {"training_loss": training_loss.avg}, epoch + 1)


# Part 4. 'main' function
if __name__ == '__main__':
    logger.info("training status: ")
    for epoch in range(basic_configs['num_epochs']):
        logger.info("Begin training epoch {}".format(epoch + 1))
        train_function(epoch)

        net_checkpoint_name = args.exp + "_net_epoch" + str(epoch + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": epoch + 1,
                     "network": net.module.state_dict()}
        torch.save(net_state, net_checkpoint_path)
