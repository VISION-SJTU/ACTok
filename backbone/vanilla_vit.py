import torch
import timm
# from timm import list_models

class VanillaViT(torch.nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, model_name="vit_base_patch16_224", pretrained=False, img_size=224):
        super(VanillaViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # Positional embeddings are explicitly interpolated for PCA resolutions.
        self.model.patch_embed.strict_img_size = False
        self.intep_size = img_size // 16
        # self.interp_pos_embed()
    
    def interp_pos_embed(self, new_size=None, mode='bicubic'):
        b, L, d = self.model.pos_embed.shape
        cls_token = self.model.pos_embed[:, :1, :]    # keep [CLS] token
        pos_tokens = self.model.pos_embed[:, 1:, :]   # remove [CLS] token
        
        # infer original grid size
        orig_size = int(L**0.5)
        assert orig_size * orig_size == L - 1, "pos_embed has unexpected shape"
        # target size
        if new_size is None:
            new_size = (self.intep_size, self.intep_size)
        elif isinstance(new_size, int):
            new_size = (new_size, new_size)
        # reshape to [B, C, H, W]
        pos_tokens = pos_tokens.view(b, orig_size, orig_size, d).permute(0, 3, 1, 2)
        # interpolate
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_size, mode=mode, align_corners=False)
        # back to [B, HW, C]
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).view(b, -1, d)
        # concat cls + patch tokens
        self.model.pos_embed = torch.nn.Parameter(torch.cat([cls_token, pos_tokens], dim=1))
    
    def forward(self, x):
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

# def load_vit_model(model_name, pretrained=True, num_classes=1000):
#     try:
#         model = timm.create_model(
#             model_name, 
#             pretrained=pretrained, 
#             num_classes=num_classes
#         )
#         return model
#     except Exception as e:
#         print(f"加载模型 {model_name} 失败: {e}")
#         return None

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