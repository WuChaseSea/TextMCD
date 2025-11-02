import os
import numpy as np
import random
import torchsampler
import torch
from torch.utils.data import DataLoader

from src.datasets import datasets as registry

from src.utils.setup_seed import setup_seed_obj
setup_seed_obj.setup_seed(False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data(cfg, seed=None):
    data_type = cfg.type 
    fold = cfg.get("fold", 0)
    num_folds = cfg.get("num_folds", 5)
    batch_size = cfg.get("batch_size", 32) 
    num_classes = cfg.get("num_classes", 2)
    stratified_by = cfg.get("stratified_by", None) 
    group_by = cfg.get("group_by", None)
    dataset_cfg = cfg.get("dataset", {})

    seed = setup_seed_obj.seed
    g = torch.Generator()
    g.manual_seed(seed)
    
    if data_type is not None:
        DataClass = registry[data_type]
    else:
        raise NotImplementedError()

    if cfg.data_path or cfg.train_data_path:
        train_data_path = cfg.train_data_path if cfg.train_data_path else os.path.join(cfg.data_path, 'images')
        train_label_path = cfg.train_label_path if cfg.train_label_path else os.path.join(cfg.data_path, 'labels')
    else:
        train_data_path = None
        train_label_path = None
    if cfg.data_path or cfg.val_data_path:
        val_data_path = cfg.val_data_path if cfg.val_data_path else os.path.join(cfg.data_path, 'images')
        val_label_path = cfg.val_label_path if cfg.val_label_path else os.path.join(cfg.data_path, 'labels')
    else:
        val_data_path = None
        val_label_path = None
    df_train = DataClass.prepare((train_data_path, train_label_path), cfg.train_list, cfg.img_suffix, cfg.label_suffix, num_classes=num_classes, mode='train', **dataset_cfg)
    df_val = DataClass.prepare((val_data_path, val_label_path), cfg.val_list, cfg.val_img_suffix, cfg.val_label_suffix, num_classes=num_classes, mode='val', **dataset_cfg)

    train_cfg = {
        "df": df_train.reset_index(drop = True),
        "phase": "train",
        **dataset_cfg         
    }
    valid_cfg = {
        "df": df_val.reset_index(drop = True),
        "phase": "val",
        **dataset_cfg         
    }
    ds_train = DataClass(**train_cfg)
    ds_valid = DataClass(**valid_cfg)

    sampler = torchsampler.ImbalancedDatasetSampler(ds_train) if ds_train.balance_key else None

    def dl_train(shuffle = True, drop_last = True, num_workers = cfg.get('num_workers'), sampler = sampler):
        sampler = {"sampler": sampler} if sampler else {"shuffle": shuffle}
        return DataLoader(ds_train, 
                        batch_size, 
                        pin_memory=False,
                        shuffle=True,
                        drop_last = False, 
                        num_workers = num_workers,
		                worker_init_fn = seed_worker, 
                        generator=g, persistent_workers=False)

    def dl_valid(shuffle = False, num_workers = cfg.get('num_workers')):
        return DataLoader(ds_valid, batch_size, 
                        pin_memory=False,
                        shuffle=False,
                        drop_last=False,
                        num_workers = num_workers,
                        worker_init_fn = seed_worker, 
                        generator=g, persistent_workers=False)

    return (ds_train, ds_valid), (dl_train, dl_valid)
