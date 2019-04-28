from IPython import embed

import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
import multiprocessing as mp

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob
from contextlib import closing
sys.path.append(os.getcwd())
from config import config

def round_up(value):
    return round(value + 1e-6 + 1000) - 1000

def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    im_h, im_w, _ = img.shape

    xmin = cx - (original_sz - 1) / 2.
    xmax = xmin + original_sz - 1
    ymin = cy - (original_sz - 1) / 2.
    ymax = ymin + original_sz - 1

    left = int(round_up(max(0., -xmin)))
    top = int(round_up(max(0., -ymin)))
    right = int(round_up(max(0., xmax - im_w + 1)))
    bottom = int(round_up(max(0., ymax - im_h + 1)))

    xmin = int(round_up(xmin + left))
    xmax = int(round_up(xmax + left))
    ymin = int(round_up(ymin + top))
    ymax = int(round_up(ymax + top))
    r, c, k = img.shape
    if any([top, bottom, left, right]):
        # 0 is better than 1 initialization
        te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
        te_im[top:top + r, left:left + c, :] = img
        if top:
            te_im[0:top, left:left + c, :] = img_mean
        if bottom:
            te_im[r + top:, left:left + c, :] = img_mean
        if left:
            te_im[:, 0:left, :] = img_mean
        if right:
            te_im[:, c + left:, :] = img_mean
        im_patch_original = te_im[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch_original = img[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_sz, original_sz):
        # zzp: use cv to get a better speed
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    scale = float(model_sz) / im_patch_original.shape[0]
    return im_patch, scale
def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    scale_z = float(size_z) / s_z

    s_x = s_z * size_x / float(size_z)
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x
    # point_1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
    # point_2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2
    # frame = cv2.rectangle(instance_img, (int(point_1[0]),int(point_1[1])), (int(point_2[0]),int(point_2[1])), (0, 255, 0), 2)
    # cv2.imwrite('1.jpg', frame)
    return instance_img, w_x, h_x, scale_x


def worker(output_dir, video_dir):
    if 'YT-BB' in video_dir:
        image_names = glob(os.path.join(video_dir, '*.jpg'))
        image_names = sorted(image_names, key=lambda x: int(
            x.split('/')[-1].split('_')[1]))
        video_name = video_dir.split('/')[-1]
        save_folder = os.path.join(output_dir, video_name)
        anno_path = '/mnt/diska1/YT-BB/xml/youtube_dection_frame_xml_temp'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        trajs = {}
        for image_name in image_names:
            img = cv2.imread(image_name)
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
            anno_name = os.path.join(anno_path, video_name, image_name.split(
                '/')[-1]).replace('.jpg', '.xml')
            tree = ET.parse(anno_name)
            root = tree.getroot()
            bboxes = []
            filename = root.find('filename').text
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                bbox = list(map(int, [bbox.find('xmin').text,
                                      bbox.find('ymin').text,
                                      bbox.find('xmax').text,
                                      bbox.find('ymax').text]))

                trkid = int(obj.find('trackid').text)
                if trkid in trajs:
                    trajs[trkid].append(filename)
                else:
                    trajs[trkid] = [filename]
                instance_crop_size = int(
                    np.ceil((config.instance_size + config.max_translate * 2) * (1 + config.scale_resize)))
                bbox = np.array(
                    [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, bbox[2] - bbox[0] + 1,
                     bbox[3] - bbox[1] + 1])

                instance_img, w, h, _ = get_instance_image(img, bbox,
                                                           config.exemplar_size, instance_crop_size,
                                                           config.context_amount,
                                                           img_mean)
                instance_img_name = os.path.join(save_folder,
                                                 filename + ".{:02d}.x_{:.2f}_{:.2f}.jpg".format(trkid, w, h))
                cv2.imwrite(instance_img_name, instance_img)
    else:
        image_names = glob(os.path.join(video_dir, '*.JPEG'))
        image_names = sorted(image_names, key=lambda x: int(
            x.split('/')[-1].split('.')[0]))
        video_name = video_dir.split('/')[-1]
        save_folder = os.path.join(output_dir, video_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        trajs = {}
        for image_name in image_names:
            img = cv2.imread(image_name)
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
            anno_name = image_name.replace('Data', 'Annotations')
            anno_name = anno_name.replace('JPEG', 'xml')
            tree = ET.parse(anno_name)
            root = tree.getroot()
            bboxes = []
            filename = root.find('filename').text
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                bbox = list(map(int, [bbox.find('xmin').text,
                                      bbox.find('ymin').text,
                                      bbox.find('xmax').text,
                                      bbox.find('ymax').text]))
                trkid = int(obj.find('trackid').text)
                if trkid in trajs:
                    trajs[trkid].append(filename)
                else:
                    trajs[trkid] = [filename]
                instance_crop_size = int(
                    np.ceil((config.instance_size + config.max_translate * 2) * (1 + config.scale_resize)))
                bbox = np.array(
                    [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, bbox[2] - bbox[0] + 1,
                     bbox[3] - bbox[1] + 1])

                instance_img, w, h, _ = get_instance_image(img, bbox,
                                                           config.exemplar_size, instance_crop_size,
                                                           config.context_amount,
                                                           img_mean)
                instance_img_name = os.path.join(save_folder,
                                                 filename + ".{:02d}.x_{:.2f}_{:.2f}.jpg".format(trkid, w, h))
                cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs


def processing(vid_dir, output_dir, num_threads=mp.cpu_count()):
    # get all 4417 videos in vid and all video in ytbb
    vid_video_dir = os.path.join(vid_dir, 'Data/VID')
    #ytb_video_dir = ytb_dir
    all_videos = glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
        glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
        glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
        glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
        glob(os.path.join(vid_video_dir, 'val/*'))

    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #functools.partial(worker, output_dir)(all_videos[5333])
    with closing(Pool(processes=num_threads)) as pool:
        for ret in tqdm(pool.imap_unordered(
                functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            meta_data.append(ret)
        pool.terminate()
    # save meta data
    pickle.dump(meta_data, open(os.path.join(
        output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    Fire(processing)
