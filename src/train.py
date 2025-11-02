import os
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, StochasticWeightAveraging, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch

torch.backends.cudnn.enabled = False
import cv2

cv2.setNumThreads(4)

def get_trainer(args, cfg):
    # csv logger
    log_type = cfg.train.pop('log_type')
    log_save_dir = cfg.train.pop('save_folder')
    if log_type == 'csv':
        logger = [
            CSVLogger(log_save_dir,
                      name=cfg.name,
                      version=cfg.version,
                      flush_logs_every_n_steps=cfg.train.log_step),
        ]
    elif log_type == 'tb':
        # tensorboard logger
        logger = [
            TensorBoardLogger(log_save_dir,
                              name=cfg.name,
                              version=cfg.version,
                              ),
        ]
    else:
        raise ValueError("Unknown logger type: {}".format(log_type))

    # callbacks
    monitor = cfg.train.get("monitor", "valid_metric")
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_save_dir, cfg.name, cfg.version),
            filename='{epoch}_{' + monitor + ':.3f}',
            save_last=True,
            save_top_k=10,
            save_weights_only=False,
            mode=cfg.train.get('monitor_mode', 'max'),
            monitor=monitor),
        RichProgressBar(leave=True),
        LearningRateMonitor('step')
    ]
    if cfg.train.get("swa", False):
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.train.get("swa_lrs", 1e-4), swa_epoch_start=0.7))

    # trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=len(args.gpus.split(",")),
        precision="16-mixed" if cfg.train.get("precision") == 16 else 32,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=cfg.train.num_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_step,
        val_check_interval=cfg.train.get('val_interval_step', None),
        check_val_every_n_epoch=cfg.train.val_interval,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps
    )
    return trainer
