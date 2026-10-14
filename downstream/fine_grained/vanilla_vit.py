import torch
import timm
from math import sqrt
import torch.nn as nn
# from timm import list_models

class VanillaViT(torch.nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super(VanillaViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
    
    def forward(self, x):
        if self.eval():
            # use cls_token interpolation
            b, L, d = self.model.pos_embed.shape # [b, l, d]
            L = L - 1 # not cls token
            h = x.shape[-2] // (224 // int(sqrt(L)))
            if (h**2) != L:
                pos_embed = self.model.pos_embed[:, 1:, :]
                pos_embed = pos_embed.view(b, int(sqrt(L)), int(sqrt(L)), d).permute(0, 3, 1, 2)  # [b, c, h, w]
                pos_embed = nn.functional.interpolate(pos_embed, size=(h, h), mode='bilinear', align_corners=True)
                pos_embed = pos_embed.permute(0, 2, 3, 1).view(b, -1, d)  # [b, h*w, c]
                self.model.pos_embed = nn.Parameter((torch.cat([self.model.pos_embed[:, [0], :], pos_embed], dim=1)))
        B = x.shape[0]
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        # x = x[:,0,:]
        # x = self.model.head(x)
        return x

def load_vit_model(model_name, pretrained=True, num_classes=1000):
    try:
        model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        return model
    except Exception as e:
        print(f"加载模型 {model_name} 失败: {e}")
        return None

# # 加载ViT-Base和ViT-Large
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ViT-Base模型
# vit_base = load_vit_model('vit_base_patch16_224', pretrained=True)
# if vit_base:
#     vit_base = vit_base.to(device)
#     vit_base.eval()
#     print(f"ViT-Base加载成功! 参数量: {sum(p.numel() for p in vit_base.parameters()):,}")

# myvit = VanillaViT("vit_base_patch16_224")
# myvit = myvit.to(device)
# myvit.eval()

# x = torch.rand(1, 3, 224, 224)
# x = x.to(device)
# print(dir(vit_base), vit_base.__class__)
# with torch.no_grad():
#     # out = vit_base(x)
#     # print(out.shape)

#     out2 = myvit(x)
#     print(out2.shape)

#     # diff = out - out2
#     # print(diff)