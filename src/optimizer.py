import torch
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    CosineAnnealingLR)
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

def get_optimizer(self, cfg):
    cfg = cfg.copy()
    opt_type = cfg.pop("optimizer")
    sch_type = cfg.pop("scheduler")
    optimizer = eval("get_" + opt_type)(self, cfg)
    scheduler = eval("get_" + sch_type)(self, cfg, optimizer)
    return optimizer, scheduler


def get_adam(self, cfg):
    optimizer = torch.optim.AdamW(
        self.model.parameters(), 
        lr = cfg.learning_rate, 
        weight_decay = cfg.weight_decay,
        eps=1e-5)
    return optimizer

def get_sgd(self, cfg):
    optimizer = torch.optim.SGD(
        self.model.parameters(), 
        lr = cfg.learning_rate, 
        weight_decay = cfg.weight_decay)
    return optimizer
        
def get_one_cycle(self, cfg, optimizer):
    steps_per_epoch = int(len(self.train_dataloader()))
    scheduler = {
        'scheduler': OneCycleLR(
            optimizer, 
            max_lr = cfg.learning_rate, 
            steps_per_epoch = steps_per_epoch, 
            epochs = cfg.num_epochs, 
            anneal_strategy = "cos", 
            final_div_factor = 1/cfg['gamma'],), 
        'name': 'learning_rate', 
        'interval':'step', 
        'frequency': 1
    }
    return scheduler

def get_exp(self, cfg, optimizer):
    """ExponentialLR scheduler"""
    scheduler = {
        'scheduler': ExponentialLR(
            optimizer, 
            gamma = cfg.gamma, ),
        'name': 'learning_rate', 
        'interval':'epoch', 
        'frequency': 1
    }
    return scheduler

def get_cos_iterwise(self, cfg, optimizer):
    steps_per_epoch = int(len(self.train_dataloader()))
    scheduler = {
        'scheduler': OneCycleLR(
            optimizer, 
            pct_start = 0.1, 
            cycle_momentum = False,
            max_lr = cfg.learning_rate, 
            steps_per_epoch = steps_per_epoch, 
            epochs = cfg.num_epochs, 
            anneal_strategy = "cos", 
            three_phase = False), 
        'name': 'learning_rate', 
        'interval':'step', 
        'frequency': 1
    }
    return scheduler

def get_cos(self, cfg, optimizer):
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=cfg.learning_rate*0.3)
    scheduler = {
        'scheduler': scheduler, 
        'name': 'learning_rate', 
        'interval':'epoch', 
        'frequency': 1
    }
    return scheduler

def get_cos_restart(self, cfg, optimizer):
    steps_per_epoch = int(len(self.train_dataloader()))
    scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = cfg.restart_epoch * steps_per_epoch,
            T_mult = 1,
            eta_min = 1e-5,
        )
    scheduler = {
        'scheduler': scheduler, 
        'name': 'learning_rate', 
        'interval':'step', 
        'frequency': 1
    }
    return scheduler

def get_steplr(self, cfg, optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        cfg.step_size,
        cfg.gamma
    )

    return scheduler
