import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.thirdmodels import clip

import timm


class CompareModel(nn.Module):
    def __init__(self,
                name):
        super().__init__()
        self.name = name
        if name == "MGCR":
            from src.models.compare_models.MGCR.nets.network import TCSI

            self.model = TCSI()
        elif name == "MMChange":
            from src.models.compare_models.MMChange.model import BaseNet

            self.model = BaseNet(3, 1)
            self.model_clip, _ = clip.load("./pretrained_models/clip/ViT-B-32.pt")

    def forward(self, img):
        if self.name == "MGCR":
            x, text_token_a, text_token_b, text_token_mask_a, text_token_mask_b = img
            x_a = x[:, :3, :, :]
            x_b = x[:, 3:, :, :]
            try:
                if text_token_a.shape[0] != 1:
                    out = self.model(x_a, x_b, text_token_a.squeeze(), text_token_b.squeeze(), text_token_mask_a.squeeze(), text_token_mask_b.squeeze())
                else:
                    out = self.model(x_a, x_b, text_token_a[:, 0, :], text_token_b[:, 0, :], text_token_mask_a[:, 0, :], text_token_mask_b[:, 0, :])
            except Exception as e:
                print(f"exception: {e}")
                import pdb;pdb.set_trace()
            return out[0]
        elif self.name == "MMChange":
            x, text_token_a, text_token_b, target_var = img
            x_a = x[:, :3, :, :]
            x_b = x[:, 3:, :, :]
            text_features_a = self.model_clip.encode_text(text_token_a[:,0,:])
            text_features_b = self.model_clip.encode_text(text_token_b[:,0,:])
            output = self.model(x_a, x_b, text_features_a.float(), text_features_b.float())
            def BCEDiceLoss(inputs, targets):
                # print(inputs.shape, targets.shape)
                bce = F.binary_cross_entropy(inputs, targets)
                inter = (inputs * targets).sum()
                eps = 1e-5
                dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
                # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
                return bce + 1 - dice
            
            # loss = BCEDiceLoss(output, target_var.float()) + BCEDiceLoss(output2, target_var.float()) + BCEDiceLoss(output3, target_var.float()) + \
            #    BCEDiceLoss(output4, target_var.float())
            return output
        out = self.model(img)
        return out
