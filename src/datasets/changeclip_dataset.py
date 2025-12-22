import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import inspect

from torch.utils.data import Dataset
from .base_dataset import BaseData
from .builder import build_trans
from torchvision import transforms as T


class ChangeClipData(BaseData):
    def __init__(self, df, phase, trans=None, resample_query={}, balance_key=None, num_classes=2, **dataset_cfg):
        super().__init__(df, phase, trans, resample_query, balance_key, num_classes, **dataset_cfg)
        self.df = df
        self.phase = phase
        self.trans = build_trans(trans.get(self.phase, None), trans.get("type", "A"))
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
            self.value_df.append(query_df.reset_index(drop=True))
        self.df = pd.concat([self.df] + self.key_df).reset_index(drop=True)

        self.balance_key = balance_key if phase == "train" else None
        normalize = T.Normalize(mean=dataset_cfg['mean_value'], std=dataset_cfg['std_value'])
        self.norm = T.Compose([
            T.ToTensor(),
            normalize
        ])
        self.band_num = dataset_cfg['band_num']
        json_time1 = Path(self.df.iloc[0]['image_file_A']).parent.parent / 'A_clipcls_56_new_vit16.json'
        json_time2 = Path(self.df.iloc[0]['image_file_A']).parent.parent / 'B_clipcls_56_new_vit16.json'
        # json_time1 = Path(self.df.iloc[0]['image_file_A']).parent.parent / 'A_clipcls_56_vit16.json'
        # json_time2 = Path(self.df.iloc[0]['image_file_A']).parent.parent / 'B_clipcls_56_vit16.json'
        self.json_data1 = self.load_json(str(json_time1))
        self.json_data2 = self.load_json(str(json_time2))

    @staticmethod
    def prepare_deprecated(data_dir, list_file=None, img_suffix=None, label_suffix=None, num_classes=2, **dataset_cfg):
        train_data_path, train_label_path = data_dir
        train_images_A = sorted(glob.glob(os.path.join(train_data_path, f"A/*{img_suffix}")))
        train_images_B = sorted(glob.glob(os.path.join(train_data_path, f"B/*{img_suffix}")))
        # train_labels = sorted(glob.glob(os.path.join(train_label_path, f"*{label_suffix}")))
        train_labels = [os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix) for i in
                        train_images_A if os.path.exists(
                os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix))]

        df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B, "mask_file": train_labels})
        if list_file is not None:
            def load_data_with_list(list_file, full_fn_list, train_path):
                sub_fn_list = []
                sub_ids = []
                with open(list_file, 'r') as f:
                    for line in f.readlines():
                        sub_fn_list.append(os.path.splitext(line.strip())[0])
                for idx, fn in enumerate(full_fn_list):
                    fbn = os.path.splitext(os.path.basename(fn))[0]
                    if fbn in sub_fn_list:
                        sub_ids.append(idx)
                return sub_ids

            sub_ids = load_data_with_list(list_file, train_images_A, train_data_path)
            df = df.iloc[sub_ids]
        return df

    @staticmethod
    def prepare(data_dir, list_file=None, img_suffix=None, label_suffix=None, num_classes=2, **dataset_cfg):
        """
        stack = inspect.stack()
        caller_frame = stack[2]
        caller_frame_name = caller_frame[3]
        print(f'调用函数名：{caller_frame_name}')
        """
        train_data_path, train_label_path = data_dir
        train_images_A, train_images_B, train_labels = [], [], []
        if list_file is None:
            train_images_A = sorted(glob.glob(os.path.join(train_data_path, f"A/*{img_suffix}")))
            train_images_B = sorted(glob.glob(os.path.join(train_data_path, f"B/*{img_suffix}")))
            train_labels = [os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix) for
                            i in train_images_A if os.path.exists(
                    os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix))]
        else:
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f'Data Statisticsing'):
                    line = Path(line.strip())
                    if len(line.parts) == 1:
                        train_images_A.append(str(Path(train_data_path) / 'A' / (line.stem + img_suffix)))
                        train_images_B.append(str(Path(train_data_path) / 'B' / (line.stem + img_suffix)))
                        train_labels.append(str(Path(train_label_path) / (line.stem + label_suffix)))
                    else:
                        if line.exists():
                            line_parent = line.parent.parent
                        else:
                            line_parent = line.parent
                        train_images_A.append(str(line_parent / 'A' / (line.stem + img_suffix)))
                        train_images_B.append(str(line_parent / 'B' / (line.stem + img_suffix)))
                        train_labels.append(str(line_parent / 'label' / (line.stem + label_suffix)))

        df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B, "mask_file": train_labels})
        return df

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        struct_json = {}
        for item in data:
            try:
                struct_json[Path(item['image_path']).name] = ", ".join(list(item.keys())[1:10])
            except:
                # import pdb;pdb.set_trace()
                struct_json[item] = ", ".join(data[item])
        return struct_json

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        if row["redirect"] != -1:
            idx = np.random.randint(len(self.value_df[row["redirect"]]))
            row = self.value_df[row["redirect"]].loc[idx]

        image_file_A = row["image_file_A"]
        image_file_B = row["image_file_B"]
        mask_file = row["mask_file"]
        # imgA = cv2.imread(image_file_A)
        imgA = cv2.imdecode(np.fromfile(image_file_A, dtype=np.uint8), -1)
        # imgB = cv2.imread(image_file_B)
        imgB = cv2.imdecode(np.fromfile(image_file_B, dtype=np.uint8), -1)
        try:
            imgA, imgB = [cv2.cvtColor(_, cv2.COLOR_BGR2RGB) for _ in (imgA, imgB)]
        except:
            print(f'出错了：')
            print(f'image_file_A: {image_file_A}')
            print(f'image_file_B: {image_file_B}')
            print(imgA)
            print(imgB)
        # mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        mask = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), -1)
        if mask.max() == 255:
            mask = mask / 255
        if self.trans is not None:
            aug = self.trans(image=imgA, imageB=imgB, mask=mask)
            imgA = aug['image']
            imgB = aug['imageB']
            mask = aug['mask']

        img = np.concatenate([imgA, imgB], axis=2)
        img_ori = img
        img = self.norm(img)
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
        return [img, self.json_data1[Path(image_file_A).name], self.json_data2[Path(image_file_B).name]], torch.Tensor(mask).long(), img_ori, os.path.basename(mask_file)

    def __len__(self):
        return self.df.shape[0]
