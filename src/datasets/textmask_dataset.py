import os
from pathlib import Path
import cv2
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F
from .base_dataset import BaseData
from .builder import build_trans
from .Augmentor import Augmentor, style_transfer
# from .copy_paste import copy_paste
from .DataUtils import DataUtils
from src.datasets.config import cfg
import random
import time
from PIL import Image
from copy import deepcopy
from tqdm import tqdm


class TextMaskData(BaseData):
    def __init__(self, df, phase, 
            trans = None, 
            resample_query = {},
            balance_key = None,
            num_classes = 2,
            **dataset_cfg):
        self.utils = DataUtils()
        self.mode = phase
        self.df = df
        self.phase = phase
        self.trans = build_trans(trans.get(self.phase, None), trans.get("type", "A")) if trans else None
        self.df["redirect"] = -1
        self.key_df = []
        self.value_df = []
        self.num_classes = num_classes
        for i, query in enumerate(resample_query.get(phase, [])):
            q = query.query
            r = query.ratio
            query_df = self.df.query(q).copy()
            query_df["redirect"] = i

            self.df = self.df[~self.df.index.isin(query_df.index)]

            length = int(round(len(query_df) * r))
            self.key_df.append(query_df.iloc[:length])
            self.value_df.append(query_df.reset_index(drop = True))
        self.df = pd.concat([self.df] + self.key_df).reset_index(drop = True)
        aug_params = {'aug_p': 0.5, 'ssr': False, 'distortion': True, 'hsv': False, 'noise': True, 'fancypca': True, 'clahe': False, 'degradation': False, 'addhaze': False, 'exchangeT': False}
        self.aug_toolkit = Augmentor(aug_params)
        self.balance_key = balance_key if phase == "train" else None
        normalize = T.Normalize(mean=dataset_cfg['mean_value'], std=dataset_cfg['std_value'])
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])
        self.mean_value = [0.31275325336310233, 0.3972738943025338, 0.3070039872257214]
        self.std_value = [0.12348501077068147, 0.12272028104760302, 0.14730654056310574]

    @staticmethod
    def prepare(data_dir, list_file=None, img_suffix=None, label_suffix=None, num_classes=2, **dataset_cfg):
        """
        stack = inspect.stack()
        caller_frame = stack[2]
        caller_frame_name = caller_frame[3]
        print(f'调用函数名：{caller_frame_name}')
        """
        train_data_path, train_label_path = data_dir
        pre_name, post_name = 'A', 'B'
        if '01初赛train' in train_data_path:
            pre_name = 'img1_img2_style'
            post_name = 'img2'
        train_images_A, train_images_B, train_labels = [], [], []
        if list_file is None:
            train_images_A = sorted(glob.glob(os.path.join(train_data_path, f"{pre_name}/*{img_suffix}")))
            train_images_B = sorted(glob.glob(os.path.join(train_data_path, f"{post_name}/*{img_suffix}")))
            train_labels = [os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix) for
                            i in train_images_A if os.path.exists(
                    os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix))]
        else:
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f'Data Statisticsing'):
                    line = Path(line.strip())
                    if len(line.parts) == 1:
                        train_images_A.append(str(Path(train_data_path) / f"{pre_name}" / (line.stem + img_suffix)))
                        train_images_B.append(str(Path(train_data_path) / f"{post_name}" / (line.stem + img_suffix)))
                        train_labels.append(str(Path(train_label_path) / (line.stem + label_suffix)))
                    else:
                        if line.exists():
                            line_parent = line.parent.parent
                        else:
                            line_parent = line.parent
                        train_images_A.append(str(line_parent / f"{pre_name}" / (line.stem + img_suffix)))
                        train_images_B.append(str(line_parent / f"{post_name}" / (line.stem + img_suffix)))
                        train_labels.append(str(line_parent / 'label_inst' / (line.stem + label_suffix)))
        
        df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B, "mask_file": train_labels})
        return df

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        image_file_A = row["image_file_A"]
        image_file_B = row["image_file_B"]
        mask_file = row["mask_file"]
        bbox_file = os.path.splitext(mask_file)[0] + ".txt"
        npy_data = None
        if self.mode == 'train':
            npy_file = os.path.splitext(mask_file)[0] + ".npy"
            if Path(npy_file).exists():
                npy_data = np.load(npy_file)
                npy_data = npy_data.astype(np.float32)
                tmp_num = npy_data.shape[0]
            else:
                npy_data = None

        img, gt, name, bbox, npy_data = self.load_data(image_file_A, image_file_B, mask_file, bbox_file, npy_data)
        # print(f'read data from {name}')
        img_ori = deepcopy(img)
        # img, gt, bbox = self.data_augmentation(img, gt, bbox)
        # if not self.check_numpy(gt):
        #     import ipdb;ipdb.set_trace()
        if self.mode == 'train':
            if npy_data is not None:
                if npy_data.shape[0] <= 300:
                    zero_data = np.zeros((300 - npy_data.shape[0], npy_data.shape[1]), npy_data.dtype)
                    npy_data = np.concatenate((npy_data, zero_data), axis=0)
                else:
                    npy_data = npy_data[0:300, ...]
            img, gt = self.transform(img, gt, name, npy_data)

            # if int(gt[1].max()) == 0:
            #     print(f'gt max: {gt[1].max()}, {mask_file}')
            # if gt[1].max() * 2 + 2 != tmp_num:
            #     import ipdb;ipdb.set_trace()
        else:
            img, gt = self.transform(img, gt, name, bbox)
        # img = self.norm(img)
        return img, gt, img_ori, os.path.basename(mask_file)
    
    def load_data(self, pre_img_filename, post_img_filename, gt_filename, bbox_filename, npy_data=None):
        # pre_img = self.utils.gdal_to_numpy(pre_img_filename)
        pre_img = Image.open(pre_img_filename)
        if pre_img.size != (512, 512):
            pre_img = pre_img.resize((512, 512), Image.BILINEAR)
            pre_img = np.array(pre_img)
        # post_img = self.utils.gdal_to_numpy(post_img_filename)
        post_img = Image.open(post_img_filename)
        if post_img.size != (512, 512):
            post_img = post_img.resize((512, 512), Image.BILINEAR)
            post_img = np.array(post_img)
        if self.mode != 'test':
            # gt = self.utils.gdal_to_numpy(gt_filename)[:,:,0]
            gt = np.array(Image.open(gt_filename))
            bbox = self.load_bbox(bbox_filename)
        else:
            gt= gt_filename
            bbox= None
        # if self.trans is not None:
        #     aug = self.trans(image=pre_img, imageB=post_img)
        #     pre_img = aug['image']
        #     post_img = aug['imageB']
        # gt_num = gt.max()
        # gt = cv2.resize(gt, dsize=(pre_img.shape[0], pre_img.shape[1]), interpolation=cv2.INTER_NEAREST)
        # gt = gt.astype(np.uint8)
        img = np.append(pre_img, post_img, axis=2)
        
        return img, gt, os.path.basename(gt_filename), bbox, npy_data
    
    def load_bbox(self, bbox_file):
        bbox_list = []
        if not Path(bbox_file).exists():
            return None
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if int(line.split(',')[-1]) == 1:
                    continue
                bbox_list.append([int(s) for s in line.split(',')])
        if len(bbox_list) == 0:
            return None
        return np.array(bbox_list)

    def data_augmentation(self, img, gt, bbox=None):
            
        if self.aug_toolkit is not None:
            if self.mode == 'train':
                img, gt, bbox = self.aug_toolkit(img, gt, bbox)
        return img, gt, bbox
    
    def check_numpy(self, inst):
        num = inst.max()
        num_list = np.unique(inst)
        for i in range(num):
            if i not in num_list:
                return False
        return True
    
    def transform(self, img, gt, name, bbox=None):
        img = self.transforms(np.ascontiguousarray(img, dtype = np.uint8))
        if bbox is None:
            bbox = torch.zeros(gt.shape)
        if self.mode != 'test':
            gt_inst = torch.Tensor(np.ascontiguousarray(gt, dtype = np.float32))[None]
            gt_binary = torch.zeros(gt_inst.shape)
            gt_binary[gt_inst!=0] = 1
            if self.mode == 'train':
                gts = [gt_binary, gt_inst, bbox]
            else:
                bbox_mask = self.box_to_mask(bbox, gt_inst.shape[1:])

                gts = [gt_binary, gt_inst, bbox_mask]

            if self.mode == 'val':
                gts.append(name)
        else:
            gts = [gt]
        return img, gts
        
    def box_to_mask(self, bboxes, img_size=(512, 512)):
        """
        convert bbox array to bbox mask. the mask is a (nc, h, w) tensor with the nc set to max possible instance counts.
        bbox: a list of tuples store the bbox information. Example: '[(261.0, 41.0, 32.0, 12.0, 377), (xmin, ymin, width, height, area)]'
        """
        masks = torch.zeros(64, img_size[0], img_size[1], dtype=torch.uint8) # max instance counts set to 64
        return masks
