# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

# ======== 各模态 backbone 封装 ========
class visible_module(nn.Module):
    def __init__(self, model_name="/16T/yzc/dinov2-base", freeze_backbone=True, use_lora=False):
        super(visible_module, self).__init__()
        backbone = AutoModel.from_pretrained(model_name)

        #for name, module in backbone.named_modules():
        #    if isinstance(module, torch.nn.Linear):
        #        print(name)

        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"],  # ViT 的注意力层
                lora_dropout=0.1,
                bias="none"
            )

        #backbone = get_peft_model(backbone, lora_config)

        #if freeze_backbone:
        #    for p in self.backbone.parameters():
        #        p.requires_grad = False

        if freeze_backbone:
            for name, p in backbone.named_parameters():
                if "lora" not in name:  # 只训练 LoRA
                    p.requires_grad = False

        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        feat = out.last_hidden_state[:, 0]  # CLS token
        return feat


class thermal_module(nn.Module):
    def __init__(self, model_name="/16T/yzc/dinov2-base", freeze_backbone=True, use_lora=False):
        super(thermal_module, self).__init__()
        #self.backbone = AutoModel.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)

        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"],  # ViT 的注意力层
                lora_dropout=0.1,
                bias="none"
            )

        #backbone = get_peft_model(backbone, lora_config)

        #if freeze_backbone:
        #    for p in self.backbone.parameters():
        #        p.requires_grad = False
        if freeze_backbone:
            for name, p in backbone.named_parameters():
                if "lora" not in name:  # 只训练 LoRA
                    p.requires_grad = False

        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        feat = out.last_hidden_state[:, 0]
        return feat


class depth_module(nn.Module):
    def __init__(self, model_name="/16T/yzc/dinov2-base", freeze_backbone=True, use_lora=False):
        super(depth_module, self).__init__()
        #self.backbone = AutoModel.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)

        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"],  # ViT 的注意力层
                lora_dropout=0.1,
                bias="none"
            )

        #backbone = get_peft_model(backbone, lora_config)

        #if freeze_backbone:
        #    for p in self.backbone.parameters():
        #        p.requires_grad = False

        if freeze_backbone:
            for name, p in backbone.named_parameters():
                if "lora" not in name:  # 只训练 LoRA
                    p.requires_grad = False

        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        feat = out.last_hidden_state[:, 0]
        return feat

class fusion_module(nn.Module):
    def __init__(self, model_name="/16T/yzc/dinov2-base", freeze_backbone=True, use_lora=False):
        super(fusion_module, self).__init__()
        #self.backbone = AutoModel.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)

        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"],  # ViT 的注意力层
                lora_dropout=0.1,
                bias="none"
            )

        self.fuse_stem = nn.Sequential(
            nn.Conv2d(9, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        #backbone = get_peft_model(backbone, lora_config)

        #if freeze_backbone:
        #    for p in self.backbone.parameters():
        #        p.requires_grad = False

        if freeze_backbone:
            for name, p in backbone.named_parameters():
                if "lora" not in name:  # 只训练 LoRA
                    p.requires_grad = False

        self.backbone = backbone

    def forward(self, x):
        x = self.fuse_stem(x)
        out = self.backbone(x)
        feat = out.last_hidden_state[:, 0]
        return feat

class LinearReduce(nn.Module):
    def __init__(self, D, with_norm=True, drop=0.0):
        super().__init__()
        self.proj = nn.Linear(3*D, D, bias=True)
        self.norm = nn.LayerNorm(D) if with_norm else nn.Identity()
        self.drop = nn.Dropout(drop)
    def forward(self, x):              # x: [B, 3D]
        y = self.proj(x)               # [B, D]
        y = self.drop(self.norm(y))
        return y

class ResidualAdapter(nn.Module):
    """D -> r -> D, with residual: p = x + adapter(x)."""
    def __init__(self, dim, bottleneck=128, drop=0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=True)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.up   = nn.Linear(bottleneck, dim, bias=True)
        self.norm = nn.LayerNorm(dim)

        # critical: start as near-identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.up(self.drop(self.act(self.down(x))))
        return self.norm(x + h)

# ======== 融合 + 分类网络 ========
class embed_net(nn.Module):
    def __init__(self, class_num=3, model_name="/16T/yzc/dinov2-base", freeze_backbone=True, use_lora=False):
        super(embed_net, self).__init__()
        # 三个模态 backbone
        self.visible = visible_module(model_name, freeze_backbone, use_lora)
        self.thermal = thermal_module(model_name, freeze_backbone, use_lora)
        self.depth   = depth_module(model_name, freeze_backbone, use_lora)
        self.fusion = fusion_module(model_name, freeze_backbone, use_lora)

        # 获取 backbone 输出维度
        backbone_dim = self.visible.backbone.config.hidden_size
        fusion_dim = backbone_dim * 3
        self.linear_reduce = LinearReduce(backbone_dim)

        self.adapter_v = ResidualAdapter(backbone_dim, bottleneck=128, drop=0.0)
        self.adapter_t = ResidualAdapter(backbone_dim, bottleneck=128, drop=0.0)
        self.adapter_d = ResidualAdapter(backbone_dim, bottleneck=128, drop=0.0)

        # 分类头
        self.classifier = nn.Linear(backbone_dim, class_num)
        nn.init.normal_(self.classifier.weight, std=0.01)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x_v, x_t, x_d, model=0):
        # 分别提特征
        f_v = self.visible(x_v)   # [B, D]
        f_t = self.thermal(x_t)   # [B, D]
        f_d = self.depth(x_d)     # [B, D]

        f_v = self.adapter_v(f_v)
        f_t = self.adapter_t(f_t)
        f_d = self.adapter_d(f_d)

        # 融合
        if model == 0:
            fused = torch.cat([f_v, f_t, f_d], dim=1)  # [B, 3*D]
            x = torch.cat([x_v, x_t, x_d], dim=1)
            x = self.fusion(x)
            x_miss = None
            x_ori = None
            f_ori = None
            f_miss = None
        elif model == 1:
            f_ori = f_d
            f_miss = (f_v + f_t) / 2
            fused = torch.cat([f_v, f_t, f_miss], dim=1)
            x_miss = (x_v + x_t) / 2
            x = torch.cat([x_v, x_t, x_miss], dim=1)
            x = self.fusion(x)
            x_ori = x_d
        elif model == 2:
            f_ori = f_t
            f_miss = (f_v + f_d) / 2
            fused = torch.cat([f_v, f_miss, f_d], dim=1)
            x_miss = (x_v + x_d) / 2
            x = torch.cat([x_v, x_miss, x_d], dim=1)
            x = self.fusion(x)
            x_ori = x_t
        elif model == 3:
            f_ori = f_v
            f_miss = (f_t + f_d) / 2
            fused = torch.cat([f_miss, f_t, f_d], dim=1)
            x_miss = (x_t + x_d) / 2
            x = torch.cat([x_miss, x_t, x_d], dim=1)
            x = self.fusion(x)
            x_ori = x_v

        # 分类
        fused = self.linear_reduce(fused)
        logits = self.classifier(fused)
        logits_x = self.classifier(x)
        logits_xy = self.classifier(fused + x)

        if self.training:
            return x, logits_x, fused, logits, logits_xy, x_ori, x_miss, f_ori, f_miss
        else:
            feat_norm = F.normalize(fused, p=2, dim=1)
            return feat_norm, logits_xy, logits_x, logits
