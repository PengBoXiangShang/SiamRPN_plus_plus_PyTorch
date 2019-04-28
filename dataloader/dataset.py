import torch
import cv2
import os
import sys
sys.path.append('..')
import numpy as np
import pickle
import lmdb
import hashlib
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import ipdb

from torch.utils.data.dataset import Dataset
from utils.generate_anchors import generate_anchors
from config import config
from utils.tools import box_transform, compute_iou, add_box_img, crop_and_pad

from IPython import embed


class ImagnetVIDDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data}
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.pairs_per_video_per_epoch is None or not training \
            else config.pairs_per_video_per_epoch * len(self.video_names)

        # data augmentation
        self.max_stretch = config.scale_resize
        self.max_translate = config.max_translate
        self.random_crop_size = config.instance_size
        self.center_crop_size = config.exemplar_size

        self.training = training

        #valid_scope = 2 * config.valid_scope + 1
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        config.response_map_size)
        '''
        self.anchors = generate_anchors(
            config.instance_size.config.exemplar_size, config.response_map_size)
        '''

    def imread(self, path):
        path = "." + path[15:]
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        #print img_buffer
        #print '------------------'
        #print type(img_buffer)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / (sum(weights)+0.0)

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = float(w * scale_w) / w
        scale_h = float(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, interpolation=cv2.INTER_LINEAR), gt_w, gt_h

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)

        iou = compute_iou(anchors, box).flatten()
        # print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

        # pos_index = np.random.choice(pos_index, config.num_pos)
        # neg_index = np.random.choice(neg_index, config.neg_pos)
        # max_index = np.argsort(iou.flatten())[-20:]
        # boxes = anchors[max_index]

    def __getitem__(self, idx):
        while True:
            idx = idx % len(self.video_names)
            video = self.video_names[idx]
            trajs = self.meta_data[video]
            # sample one trajs
            if len(trajs.keys()) > 0:
                break
            else:
                idx = np.random.randint(self.num)

        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        # exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))

        if 'ILSVRC2015' in video:
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video,
                                       traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
        else:
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video,
                                       traj[exemplar_idx] + ".{}.x*.jpg".format(trkid)))[0]
        exemplar_img = self.imread(exemplar_name)
        # exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        # sample instance
        if 'ILSVRC2015' in exemplar_name:
            frame_range = config.frame_range_vid
        else:
            frame_range = config.frame_range_ytb
        low_idx = max(0, exemplar_idx - frame_range)
        up_idx = min(len(traj), exemplar_idx + frame_range + 1)

        # create sample weight, if the sample are far away from center
        # the probability being choosen are high
        weights = self._sample_weights(
            exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(
            traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)

        if 'ILSVRC2015' in video:
            instance_name = glob.glob(os.path.join(
                self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
        else:
            instance_name = glob.glob(os.path.join(
                self.data_dir, video, instance + ".{}.x*.jpg".format(trkid)))[0]

        instance_img = self.imread(instance_name)
        # instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        gt_w, gt_h = float(instance_name.split(
            '_')[-2]), float(instance_name.split('_')[-1][:-4])

        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        if config.exem_stretch:
            exemplar_img, _, _ = self.RandomStretch(exemplar_img, 0, 0)
        exemplar_img, _ = crop_and_pad(exemplar_img, (exemplar_img.shape[1] - 1) / 2,
                                       (exemplar_img.shape[0] - 1) /
                                       2, self.center_crop_size,
                                       self.center_crop_size)

        # exemplar_img_np = exemplar_img.copy()

        exemplar_img = self.z_transforms(exemplar_img)

        instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
        im_h, im_w, _ = instance_img.shape
        cy_o = (im_h - 1.) / 2
        cx_o = (im_w - 1.) / 2
        cy = cy_o + \
            np.random.randint(- self.max_translate, self.max_translate + 1)
        cx = cx_o + \
            np.random.randint(- self.max_translate, self.max_translate + 1)
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy
        instance_img, scale = crop_and_pad(
            instance_img, cx, cy, self.random_crop_size, self.random_crop_size)
        instance_img = self.x_transforms(instance_img)

        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h]))))

        return exemplar_img, instance_img, regression_target, conf_target.astype(np.int64)

    def draw_img(self, img, boxes, name='1.jpg', color=(0, 255, 0)):
        # boxes (x,y,w,h)
        img = img.copy()
        img_ctx = (img.shape[1] - 1.) / 2
        img_cty = (img.shape[0] - 1.) / 2
        for box in boxes:
            point_1 = img_ctx - box[2] / 2. + \
                box[0], img_cty - box[3] / 2. + box[1]
            point_2 = img_ctx + box[2] / 2. + \
                box[0], img_cty + box[3] / 2. + box[1]
            img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                                color, 2)
        cv2.imwrite(name, img)

    def __len__(self):
        return self.num
