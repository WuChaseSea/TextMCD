# -*-coding:UTF-8 -*-
"""
* sam_optimization_application_multigpu.py
* @author wuzm
* created 2023/12/03 14:42:00
* @function: change detection application
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import torch
import sys
import yaml

sys.path.append('./')
from omegaconf import OmegaConf
from pathlib import Path

from src.tools import PredictProcess
from src.models import *

if __name__ == '__main__':
    """Inference demo for Change Detection.
    """
    import faulthandler
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./application/predict.yaml', help='config file')
    parser.add_argument('-m', '--model_path', type=str, default=None, help='model pth path')
    parser.add_argument('-pre', '--pre_img_path', type=str, default=None, help='Input image or folder')
    parser.add_argument('-post', '--post_img_path', type=str, default=None, help='Input image or folder')
    parser.add_argument('-t', '--tmp_folder_path', type=str, default=None, help='tmp save folder path')
    parser.add_argument('-o', '--output_folder_path', type=str, default=None, help='output save folder path')
    parser.add_argument('--subsize', type=int, default=0, help='subsize')
    parser.add_argument('--padding', type=int, default=0, help='padding size')
    parser.add_argument('--paired', action='store_true', help='Use Paired Images.')

    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    config_file = args.config
    if Path(config_file).exists():
        cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.to_container(cfg)
    else:
        cfg = {
            'model': {
                'model_path': None
            },
            'dataset': {
                'pre_img_path': None,
                'post_img_path': None,
                'tmp_folder_path': None,
                'output_folder_path': None,
            },
            'predict': {
                'subsize': 1024,
                'padding': 256
            }
        }

    cfg['model']['model_path'] = args.model_path if args.model_path else cfg['model']['model_path']
    cfg['dataset']['pre_img_path'] = args.pre_img_path if args.pre_img_path else cfg['dataset']['pre_img_path']
    cfg['dataset']['post_img_path'] = args.post_img_path if args.post_img_path else cfg['dataset']['post_img_path']
    cfg['dataset']['tmp_folder_path'] = args.tmp_folder_path if args.tmp_folder_path else cfg['dataset'][
        'tmp_folder_path']
    cfg['dataset']['output_folder_path'] = args.output_folder_path if args.output_folder_path else cfg['dataset'][
        'output_folder_path']
    cfg['predict']['subsize'] = args.subsize if args.subsize > 0 else cfg['predict']['subsize']
    cfg['predict']['padding'] = args.padding if args.padding > 0 else cfg['predict']['padding']
    cfg['predict']['paired'] = args.paired if args.paired else cfg['predict']['paired']

    print(yaml.dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True))

    def load_model(model_path):
        model_path_list = []
        if isinstance(model_path, str):
            model_path_list.append(model_path)
        else:
            model_path_list = [i for i in model_path]
        model_list = []
        model_path = None
        for model_path in model_path_list:
            stt = torch.load(model_path, map_location="cpu")
            hyper_parameters = stt["hyper_parameters"]
            cfg = OmegaConf.create(eval(str(stt["hyper_parameters"]))).model
            stt = {k[6:]: v for k, v in stt["state_dict"].items()}
            if 'load_from' in cfg:
                cfg.pop("load_from")
            if 'resume_checkpoint' in cfg:
                cfg.pop("resume_checkpoint")
            model = eval(cfg.pop("type"))(**cfg)
            model.load_state_dict(stt, strict=False)
            model.eval()
            model.to('cuda:0')
            model_list.append(model)
        return model_list, hyper_parameters
    
    model, hyper_parameters = load_model(model_path=cfg['model']['model_path'])
    
    # model.load()
    local_rank = args.local_rank
    print(local_rank)

    use_dist = False
    if (torch.cuda.device_count() > 1) and (local_rank != -1):
        import torch.distributed as dist
        use_dist = True
    # 预测
    app = PredictProcess(
        cfg['mode'],
        cfg['dataset'],
        cfg['predict'],
        cfg['postprocessing'],
        hyper_parameters,
        local_rank
    )
    app.process(model)
    if use_dist:
        dist.barrier()
        print(f'now dist get rank: {dist.get_rank()}')
        if dist.get_rank() == 0:
            print(f'go to post processing...')
            app.post_processing()
    else:
        app.post_processing()
