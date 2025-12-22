import os
import time
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import ml_collections

from .SGCM import SGCM, ConvTransBN
from .Vit import VisionTransformer
from .module_clip import CLIP, convert_weights, _PT_NAME
from .PVTv2 import pvt_v2_b2
from .alignment_decoder import Decoder, BIFusion


matplotlib.use('Agg')

# Text-Guided Cross-Modal Semantic Interaction for Remote Sensing Change Detection
class TCSI(nn.Module):
    def __init__(self, num_classes=1, pvt_pretrain=True,token_len=64):
        super(TCSI, self).__init__()

        # Transformer Branch with PVTv2 pretraining
        # Params choose b2
        self.backbone = pvt_v2_b2()
        if pvt_pretrain:
            path = './pretrained_models/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.reprocessing = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
        )
        self.sigmoid = nn.Sigmoid()

        self.load_clip()
        self.get_ViT_config()
        self.text_module = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.SGCM = SGCM(img_size=8, channel_num=512, patch_size=1, embed_dim=512)
        self.CTBN_t = ConvTransBN(in_channels=token_len, out_channels=8 * 8)
        self.I_ViT = VisionTransformer(img_size=8, channel_num=512, patch_size=1, embed_dim=512)

        self.BIFusion1 = BIFusion(512, 512, 512)
        self.BIFusion2 = BIFusion(320, 320, 320)
        self.BIFusion3 = BIFusion(128, 128, 128)
        self.BIFusion4 = BIFusion(64, 64, 64)
        self.Decoder1 = Decoder(512, 320, 320)
        self.Decoder2 = Decoder(320, 128, 128)
        self.Decoder3 = Decoder(128, 64, 64)

        self.re_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.re_conv2 = nn.Sequential(
            nn.Conv2d(640, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
        )
        self.re_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.re_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.final_process = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
        self.num_images = 0

    def vis_feature(self, feas):
        self.num_images += 1
        for i, f in enumerate(feas):
            f = f[0].cpu().mean(dim=0)
            # if i == 4 or i == 5:
            #     f = f.view(64, 64)
            path = f'test_image/RGB_{self.num_images}_{i}.png'
            fig, ax = plt.subplots(1, 1, tight_layout=True)
            ax.imshow(f.detach(), cmap='jet')
            ax.axis('off')
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()

    def get_ViT_config(self):
        config = ml_collections.ConfigDict()
        config.transformer = ml_collections.ConfigDict()
        config.base_channel = 64
        config.clip_backbone = "ViT-B/32"
        config.text_mask_rate = 0.3
        config.img_mask_rate = 0.5
        config.pool_mode = "max_pool"  # max_pool, aver_pool
        config.rec_trans_num_layers1 = 4
        config.mask_mode = "dist"
        config.frozen_clip = True
        config.mask_mode_dist_random = True
        config.dropout = False
        config.dropout_value = 0.3
        return config

    def load_clip(self):
        # Load the model
        config = self.get_ViT_config()
        backbone = config.clip_backbone
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16
        self.clip.load_state_dict(state_dict, strict=False)
        self.clip.float()


        if config.frozen_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

    def forward(self, A, B, text_token_A, text_token_B, text_mask_A, text_mask_B):
        """
            both A,B are bi-temporal RS image
            text_token is tensor of original text processed by clip.tokenize
            text_mask means the effective area of text_token
        """
        # text_token_A = self.clip.tokenize(text_A)
        # text_token_B = self.clip.tokenize(text_B)
        # text_mask_A = (text_token_A != 0)
        # text_mask_B = (text_token_B != 0)
        outputs = []
        text_mask_A = text_mask_A.view(-1, text_mask_A.shape[-1])
        text_mask_B = text_mask_B.view(-1, text_mask_B.shape[-1])
        cls_A, text_feat_A = self.clip.encode_text(text_token_A, return_hidden=True, mask=text_mask_A)
        cls_B, text_feat_B = self.clip.encode_text(text_token_B, return_hidden=True, mask=text_mask_B)

        text_A = self.text_module(text_feat_A.transpose(1, 2)).transpose(1, 2)
        text_A = self.CTBN_t(text_A)
        text_B = self.text_module(text_feat_B.transpose(1, 2)).transpose(1, 2)
        text_B = self.CTBN_t(text_B)


        # pvt0-(64,64,64) pvt1-(128,32,32) pvt2-(320,16,16) pvt3-(512,8,8)
        pvtA = self.backbone(A)
        pvtB = self.backbone(B)

        loss = []

        text_attnA, interact_featA = self.SGCM(pvtA[3], text_A)
        text_attnB, interact_featB = self.SGCM(pvtB[3], text_B)


        # LViT + 无SGCM
        # interact_featA, lossA = self.I_ViT(pvtA[3] , text_A)
        # interact_featB, lossB = self.I_ViT(pvtB[3] , text_B)
        # LViT + SGCM-L
        # interact_featA, lossA = self.I_ViT(pvtA[3], text_A + self.sigmoid(text_attnA))
        # interact_featB, lossB = self.I_ViT(pvtB[3], text_B + self.sigmoid(text_attnB))
        # LViT + SGCM-V
        interact_featA, lossA = self.I_ViT(pvtA[3] + self.sigmoid(interact_featA), text_A)
        interact_featB, lossB = self.I_ViT(pvtB[3] + self.sigmoid(interact_featB), text_B)

        # interact_featA, lossA = self.I_ViT(pvtA[3] + self.sigmoid(interact_featA), text_A + self.sigmoid(text_attnA))
        # interact_featB, lossB = self.I_ViT(pvtB[3] + self.sigmoid(interact_featB), text_B + self.sigmoid(text_attnB))

        loss.extend(lossA)
        loss.extend(lossB)

        # Pixel-wise Substraction
        Diff1 = torch.abs(pvtA[0] - pvtB[0])
        Diff2 = torch.abs(pvtA[1] - pvtB[1])
        Diff3 = torch.abs(pvtA[2] - pvtB[2])
        Diff4_t = torch.abs(interact_featA - interact_featB)

        # Channel-wise Concatenation
        Cat1 = torch.cat([pvtA[0], pvtB[0]], dim=1)
        Cat2 = torch.cat([pvtA[1], pvtB[1]], dim=1)
        Cat3 = torch.cat([pvtA[2], pvtB[2]], dim=1)
        Cat4_t = torch.cat([interact_featA, interact_featB], dim=1)
        Cat1 = self.re_conv4(Cat1)
        Cat2 = self.re_conv3(Cat2)
        Cat3 = self.re_conv2(Cat3)
        Cat4_t = self.re_conv1(Cat4_t)

        # 0-(512,8,8) 1-(320,16,16) 2-(128,32,32) 3-(64,64,64)
        fusion1 = self.BIFusion1(Diff4_t, Cat4_t)
        fusion2 = self.BIFusion2(Diff3, Cat3)
        fusion3 = self.BIFusion3(Diff2, Cat2)
        fusion4 = self.BIFusion4(Diff1, Cat1)

        up_1 = self.Decoder1(fusion1, fusion2)
        up_2 = self.Decoder2(up_1, fusion3)
        up_out = self.Decoder3(up_2, fusion4)

        out = F.interpolate(self.final_process(up_out), scale_factor=4, mode='bilinear')

        return out, loss
