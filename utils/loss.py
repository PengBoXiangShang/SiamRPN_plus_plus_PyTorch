import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
 cited from https://github.com/arbitularov/SiamRPN-PyTorch/blob/master/train/loss.py
'''


def rpn_cross_entropy_balance(input, target, num_pos, num_neg, ohem_pos=None, ohem_neg=None):
    r"""
    :param input: (N,1125,2)
    :param target: (15x15x5,)
    :return:
    """
    # if ohem:
    #     final_loss = rpn_cross_entropy_balance_parallel(input, target, num_pos, num_neg, anchors, ohem=True,
    #                                                     num_threads=4)
    # else:

    #print('target', target.shape[0])
    #print('target', target.shape[1])
    loss_all = []
    for batch_id in range(target.shape[0]):
        #print('batch_id', batch_id)
        min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
        min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[
                      0]) * num_neg / num_pos, num_neg))
        pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
        neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist()

        if ohem_pos:
            if len(pos_index) > 0:
                pos_loss_bid = F.cross_entropy(input=input[batch_id][pos_index],
                                               target=target[batch_id][pos_index], reduce=False)
                selected_pos_index = nms(
                    anchors[pos_index], pos_loss_bid.cpu().detach().numpy(), min_pos)
                pos_loss_bid_final = pos_loss_bid[selected_pos_index]
            else:
                print 'Feng Jin'
                pos_loss_bid = torch.FloatTensor([0])  # .cuda()
                pos_loss_bid_final = pos_loss_bid
        else:
            pos_index_random = random.sample(pos_index, min_pos)
            if len(pos_index) > 0:
                pos_loss_bid_final = F.cross_entropy(input=input[batch_id][pos_index_random],
                                                     target=target[batch_id][pos_index_random], reduce=False)
            else:
                # print 'Feng Jin'
                pos_loss_bid_final = torch.FloatTensor([0]).cuda()

        if ohem_neg:
            if len(pos_index) > 0:
                neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
                                               target=target[batch_id][neg_index], reduce=False)
                selected_neg_index = nms(
                    anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), min_neg)
                neg_loss_bid_final = neg_loss_bid[selected_neg_index]
            else:
                neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
                                               target=target[batch_id][neg_index], reduce=False)
                selected_neg_index = nms(
                    anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), num_neg)
                neg_loss_bid_final = neg_loss_bid[selected_neg_index]
        else:
            if len(pos_index) > 0:
                neg_index_random = random.sample(
                    np.where(target[batch_id].cpu() == 0)[0].tolist(), min_neg)
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduce=False)
            else:
                neg_index_random = random.sample(
                    np.where(target[batch_id].cpu() == 0)[0].tolist(), num_neg)
                neg_loss_bid_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                     target=target[batch_id][neg_index_random], reduce=False)
        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2.
        loss_all.append(loss_bid)
    final_loss = torch.stack(loss_all).mean()
    return final_loss


def rpn_smoothL1(input, target, label, num_pos=16, ohem=None):
    '''
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    '''
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos)
        if ohem:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(
                    input[batch_id][pos_index], target[batch_id][pos_index], reduce=False)
                sort_index = torch.argsort(loss_bid.mean(1))
                loss_bid_ohem = loss_bid[sort_index[-num_pos:]]
            else:
                #loss_bid_ohem = torch.FloatTensor([0]).cuda()[0]
                loss_bid_ohem = torch.FloatTensor([0])[0]
            loss_all.append(loss_bid_ohem.mean())
        else:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(
                    input[batch_id][pos_index], target[batch_id][pos_index])
            else:
                loss_bid = torch.FloatTensor([0]).cuda()[0]
                #loss_bid = torch.FloatTensor([0])[0]
            loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss
