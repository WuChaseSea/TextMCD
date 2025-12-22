import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import inspect
from PIL import Image
from osgeo import gdal
import pandas as pd

from torch.utils.data import Dataset
from .base_dataset import BaseData
from .builder import build_trans

from src.models.thirdmodels import clip


class MGCRData(BaseData):
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
        img_suffix_list = [
            '.tif',
            '.jpg',
            '.png'
        ]
        train_data_path, train_label_path = data_dir
        
        pre_name, post_name = 'A', 'B'
        if train_data_path is not None and '01初赛train' in train_data_path:
            pre_name = 'img1'
            post_name = 'img2'
        if train_data_path is not None and 'SYSUCD' in train_data_path:
            pre_name = 'time1'
            post_name = 'time2'
        train_images_A, train_images_B, train_labels = [], [], []
        if list_file is None:
            if 'SYSUCD' in train_data_path:
                train_images_A = sorted(glob.glob(os.path.join(train_data_path, f"time1/*{img_suffix}")))
                train_images_B = sorted(glob.glob(os.path.join(train_data_path, f"time2/*{img_suffix}")))
                train_labels = [os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix)
                                for
                                i in train_images_A if os.path.exists(
                        os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix))]
            else:
                train_images_A = sorted(glob.glob(os.path.join(train_data_path, f"{pre_name}/*{img_suffix}")))
                train_images_B = sorted(glob.glob(os.path.join(train_data_path, f"{post_name}/*{img_suffix}")))
                train_images_A = [i for i in train_images_A]
                train_images_B = [i for i in train_images_B]
                train_labels = [os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix) for
                                i in tqdm(train_images_A) if os.path.exists(
                        os.path.join(train_label_path, os.path.splitext(os.path.basename(i))[0] + label_suffix))]

        else:
            with open(list_file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f'Data Statisticsing'):
                    line = Path(line.strip())
                    if len(line.parts) == 1:
                        train_image_a_name = str(Path(train_data_path) / pre_name / (line.stem + img_suffix))
                        if not Path(train_image_a_name).exists():
                            for img_suffix1 in img_suffix_list:
                                train_image_a_name = str(Path(train_data_path) / pre_name / (line.stem + img_suffix1))
                                if Path(train_image_a_name).exists():
                                    continue
                        train_images_A.append(train_image_a_name)
                        train_image_b_name = str(Path(train_data_path) / post_name / (line.stem + img_suffix))
                        if not Path(train_image_b_name).exists():
                            for img_suffix1 in img_suffix_list:
                                train_image_b_name = str(Path(train_data_path) / post_name / (line.stem + img_suffix1))
                                if Path(train_image_b_name).exists():
                                    continue
                        train_images_B.append(train_image_b_name)
                        train_labels.append(str(Path(train_label_path) / (line.stem + label_suffix)))
                    else:
                        if line.exists():
                            line_parent = line.parent.parent
                        else:
                            line_parent = line.parent
                        train_image_a_name = str(line_parent / pre_name / (line.stem + img_suffix))
                        if not Path(train_image_a_name).exists():
                            for img_suffix1 in img_suffix_list:
                                train_image_a_name = str(line_parent / pre_name / (line.stem + img_suffix1))
                                if img_suffix1 == '.png':
                                    import ipdb;
                                    ipdb.set_trace()
                                if Path(train_image_a_name).exists():
                                    break
                        train_images_A.append(train_image_a_name)
                        train_image_b_name = str(line_parent / post_name / (line.stem + img_suffix))
                        if not Path(train_image_b_name).exists():
                            for img_suffix1 in img_suffix_list:
                                train_image_b_name = str(line_parent / post_name / (line.stem + img_suffix1))
                                if Path(train_image_b_name).exists():
                                    break
                        train_images_B.append(train_image_b_name)
                        # train_images_A.append(str(line_parent / 'A' / (line.stem + img_suffix)))
                        # train_images_B.append(str(line_parent / 'B' / (line.stem + img_suffix)))
                        train_labels.append(str(line_parent / 'label' / (line.stem + label_suffix)))
        text_a_xlsx = os.path.join(train_data_path, "A_I2T_o.xlsx")
        text_b_xlsx = os.path.join(train_data_path, "B_I2T_o.xlsx")
        df_text_a = pd.read_excel(text_a_xlsx)
        df_text_b = pd.read_excel(text_b_xlsx)
        # model, preprocess = clip.load("./pretrained_models/clip/ViT-L-14.pt")

        text_token_a = [clip.tokenize(text, context_length=64) for text in df_text_a["Description"]]
        text_token_b = [clip.tokenize(text, context_length=64) for text in df_text_b["Description"]]
        text_token_mask_a = [(text_token != 0) for text_token in text_token_a]
        text_token_mask_b = [(text_token != 0) for text_token in text_token_b]
        
        try:
            df = pd.DataFrame({"image_file_A": train_images_A, "image_file_B": train_images_B, "mask_file": train_labels, "text_token_a": text_token_a, "text_token_b": text_token_b, "text_token_mask_a": text_token_mask_a, "text_token_mask_b": text_token_mask_b})
        except:
            import pdb;pdb.set_trace()
        return df
    
    def read_mask_data(self, mask_file):
        dataset = gdal.Open(mask_file)
        band = dataset.GetRasterBand(1)
        mask = band.ReadAsArray()
        return mask

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        row = self.df.loc[idx]
        if row["redirect"] != -1:
            idx = np.random.randint(len(self.value_df[row["redirect"]]))
            row = self.value_df[row["redirect"]].loc[idx]

        image_file_A = row["image_file_A"]
        image_file_B = row["image_file_B"]
        mask_file = row["mask_file"]
        text_token_a = row["text_token_a"]
        text_token_b = row["text_token_b"]
        text_token_mask_a = row["text_token_mask_a"]
        text_token_mask_b = row["text_token_mask_b"]
        # imgA = cv2.imread(image_file_A)
        # imgA = cv2.imdecode(np.fromfile(image_file_A, dtype=np.uint8), -1)
        # imgA = np.array(Image.open(image_file_A))[:,:,::-1]
        # imgA = np.array(Image.open(image_file_A))
        # imgB = cv2.imread(image_file_B)
        # imgB = cv2.imdecode(np.fromfile(image_file_B, dtype=np.uint8), -1)
        # imgB = np.array(Image.open(image_file_B))
        # try:
        #     imgA, imgB = [cv2.cvtColor(_, cv2.COLOR_BGR2RGB) for _ in (imgA, imgB)]
        # except:
        #     print(f'出错了：')
        #     print(f'image_file_A: {image_file_A}')
        #     print(f'image_file_B: {image_file_B}')
        #     print(imgA)
        #     print(imgB)
        # mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        # mask = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), -1)
        # mask = np.array(Image.open(mask_file))
        imgA = np.array(Image.open(image_file_A))
        imgB = np.array(Image.open(image_file_B))
        mask = np.array(Image.open(mask_file))
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
        else:
            mask = mask.astype(np.uint8)
            mask = mask[:, :, 0]
            mask = mask[np.newaxis, ...]
        # print(f'image shape: {img.shape}')
        # print(f'mask shape: {mask.shape}')
        # print(f'img ori shape: {img_ori.shape}')
        return img, torch.Tensor(mask).long(), text_token_a, text_token_b, text_token_mask_a, text_token_mask_b, img_ori, os.path.basename(mask_file)

    def __len__(self):
        return self.df.shape[0]
