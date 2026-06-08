import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from resnet import resnet50, resnet18

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """从截断的正太分布中初始化张量"""
    # 使用PyTorch的内置函数
    init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, (list, tuple)):
        return (x[0], x[1])
    else:
        raise ValueError("Expected an int or a tuple/list of two ints")

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
'''
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
'''
def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        init.constant_(m.weight, 1.0)
        init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.01)
        if m.bias:
            init.zeros_(m.bias.data)

def preprocess_input(x):
    # 确保输入为224x224
    if x.size(2) != 224 or x.size(3) != 224:
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return x
'''

def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0, 0.001)  # 标准差设为 0.01
        if m.bias is not None:
            init.zeros_(m.bias.data)
'''
class SelfAttention(nn.Module):
    def __init__(self, in_channels, head_size=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.num_heads = in_channels // head_size  # 通常将通道数分配为多个头
        # Q, K, V 的线性映射
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 输出通道
        self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        #for layer in [self.query, self.key, self.value, self.fc_out]:
        #    for m in layer.modules():
        #        if isinstance(m, nn.Conv2d):
        #            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        #        if m.bias is not None:
        #            init.zeros_(m.bias.data)
    def forward(self, x):
        B, C, H, W = x.size()  # B: batch_size, C: channels, H: height, W: width

        # Step 1: 生成Q, K, V
        Q = self.query(x).view(B, self.num_heads, self.head_size, H * W).permute(0, 1, 3, 2)
        K = self.key(x).view(B, self.num_heads, self.head_size, H * W)
        V = self.value(x).view(B, self.num_heads, self.head_size, H * W).permute(0, 1, 3, 2)
        #print(852, Q.size(), K.size(), V.size())

        # Step 2: 计算注意力权重
        attention_scores = torch.matmul(Q, K) / (self.head_size ** 0.5)  # Scaled dot-product
        attention_weights = F.softmax(attention_scores, dim=-1)
        #print(963, attention_scores.shape, attention_weights.shape)

        # Step 3: 加权求和
        attention_output = torch.matmul(attention_weights, V)
        #print(attention_output.size())

        # Step 4: 合并各个头的输出并通过线性层
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(B, C, H, W)
        out = self.fc_out(attention_output)

        return out

class EnhancedFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedFusionModule, self).__init__()
        self.attention = SelfAttention(in_channels)
        self.conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        for m in [self.conv, self.bn]:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if m.bias is not None:
                init.zeros_(m.bias.data)

    def forward(self, x1, x2, x3):
        x1 = self.attention(x1)
        x2 = self.attention(x2)
        x3 = self.attention(x3)
        fused = torch.cat((x1, x2, x3), 1)
        fused = self.conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        return fused


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FusionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FusionModule, self).__init__()
        self.se = SEBlock(input_channels)
        # 使用1x1卷积来减少通道数
        self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        # BN层
        self.bn = nn.BatchNorm2d(output_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.se(x)  # 先进行注意力调整
        # 降维
        x = self.conv1x1(x)
        # 添加BN层
        x = self.bn(x)
        # 引入非线性激活
        x = self.relu(x)
        return x

class AttentionFusionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AttentionFusionModule, self).__init__()
        # 使用卷积生成注意力权重
        self.attention_conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):
        # 对每个模态的特征进行卷积，生成注意力权重
        attention1 = torch.sigmoid(self.attention_conv(x1))
        attention2 = torch.sigmoid(self.attention_conv(x2))
        attention3 = torch.sigmoid(self.attention_conv(x3))
        # 使用注意力权重加权每个模态的特征
        x1 = x1 * attention1
        x2 = x2 * attention2
        x3 = x3 * attention3
        # 融合加权后的特征
        fused = torch.cat((x1, x2), dim=1)
        fused = torch.cat((fused, x3), dim=1)
        #fused = self.conv(fused)
        return fused

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super().__init__()
        m = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # -> [B, 2048, 7, 7] (224 输入时)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super().__init__()
        m = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # -> [B, 2048, 7, 7] (224 输入时)
        return x


class depth_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super().__init__()
        m = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # -> [B, 2048, 7, 7] (224 输入时)
        return x


class TransformerBase(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True):
        super(TransformerBase, self).__init__()

        # 图像块划分与嵌入
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 添加可学习的类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 添加可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])

        # 初始化参数
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 添加类别标记和位置编码
        B = x.size(0)
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)

        return x[:, 1:]  # 移除类别标记，返回特征


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, qkv_bias=True):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features=768, hidden_features=3072, act_layer=nn.ReLU, drop=0.0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        #model_base = resnet50(pretrained=True,
        #                      last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        #model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = TransformerBase(img_size=224, patch_size=16, in_c=9, embed_dim=768, depth=12, num_heads=12)
        #self.base1 = TransformerBase(img_size=224, patch_size=16, in_c=128, embed_dim=768, depth=12, num_heads=12)

    def forward(self, x):
        x = self.base(x)
        return x

class base_resnet1(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet1, self).__init__()

        #self.base = TransformerBase(img_size=224, patch_size=16, in_c=192, embed_dim=768, depth=12, num_heads=12)
        self.base1 = TransformerBase(img_size=224, patch_size=16, in_c=192, embed_dim=768, depth=12, num_heads=12)

    def forward(self, x):
        x = self.base1(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='transformer'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.depth_module = depth_module(arch=arch)
        #self.fusion = EnhancedFusionModule(in_channels=64)
        self.base_resnet = base_resnet(arch=arch)
        self.base_resnet1 = base_resnet1(arch=arch)
        self.fusion_module = FusionModule(128, 64)
        self.fusion_module1 = FusionModule(192,64)
        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d(2048 * 3, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.non_local = no_local
        #'''
        self.non_local = no_local
        if self.non_local == 'on':
            self.NL_1 = nn.ModuleList([Non_local(768) for _ in range(3)])
            self.NL_2 = nn.ModuleList([Non_local(768) for _ in range(3)])
            self.NL_3 = nn.ModuleList([Non_local(768) for _ in range(3)])
            self.NL_4 = nn.ModuleList([Non_local(768) for _ in range(3)])


        pool_dim = 768
        #pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool



    def forward(self, x1, x2, x3, modal=0): #三模态记得modal=0,再加上
        if modal == 0:
            x1 = preprocess_input(x1)
            x2 = preprocess_input(x2)
            x3 = preprocess_input(x3)

        elif modal == 1:
            x1 = preprocess_input(x1)
            x2 = preprocess_input(x2)
            x3 = preprocess_input((x1+x2)/2)


        elif modal == 2:
            x1 = preprocess_input(x1)
            x3 = preprocess_input(x3)
            x2 = preprocess_input((x1+x3)/2)

        elif modal == 3:
            x2 = preprocess_input(x2)
            x3 = preprocess_input(x3)
            x1 = preprocess_input((x2+x3)/2)


        fused = torch.cat([x1, x2, x3], dim=1)
        x = self.base_resnet(fused)
        x = x.transpose(1, 2).reshape(x.size(0), 768, 14, 14)  # 还原回 7×7 feature map，便于复用 Non-local/GM-pool

        # shared block
        if self.non_local == 'on':
            x = self.NL_1[0](x)
            x = self.NL_2[0](x)
            x = self.NL_3[0](x)
            x = self.NL_4[0](x)

        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            # 在幂运算前，确保x_flat没有负数
            x_flat = torch.abs(x)
            x_flat = x_flat + 1e-12  # 避免数值为0，防止除以0或出现log(0)的情况
            # 使用mean函数计算均值
            mean_values = torch.mean(x_flat ** p, dim=-1)
            # 确保mean_values不为零
            mean_values = torch.where(mean_values == 0, torch.ones_like(mean_values) * 1e-12, mean_values)
            x_pool = (mean_values) ** (1 / p)
            #x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        #print("x_pool before bottleneck:", x_pool)
        feat = self.bottleneck(x_pool)
        #print("Feature before classification:", feat, feat.shape)
        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(x_pool), self.classifier(feat)