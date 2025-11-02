import torch

from src.models import models as registry

def get_model(cfg):
    cfg = cfg.copy()
    cfg.pop('resume_checkpoint')
    if cfg.get("load_from"):
        load_from = cfg.pop("load_from")
    else:
        load_from = cfg.pop("load_from")
    
    model = registry[cfg.pop("type")](**cfg)
    # print(cfg)
    if load_from:
        model_dict = model.state_dict()

        print(f'load checkpoint from {load_from}')
        stt = torch.load(load_from, map_location = "cpu")
        stt = stt["state_dict"] if "state_dict" in stt else stt
        model_stt = model.state_dict()
        if all(k.startswith("model.") and k[6:] in model_stt for k in stt.keys()):
            stt = {k[6:]: v for k, v in stt.items()}
        if all(k.startswith("module.") and k[7:] in model_stt for k in stt.keys()):
            stt = {k[7:]: v for k, v in stt.items()}
        # all_keys = stt['model'].keys()
        all_keys = stt.keys()
        new_stt = {}
        for key_name in all_keys:
            new_stt[key_name] = stt[key_name]
        model_dict.update(new_stt)

        model.load_state_dict(model_dict, strict = False)

    return model

