import torch
import torch.nn as nn


class Radio(nn.Module):
    def __init__(self, num_classes=1000):
        super(Radio, self).__init__()

        model_version = 'radio_v2.5-b'
        self.backbone = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True, force_reload=False)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    @torch.no_grad()
    def forward(self, x):
        # convert to 0-1
        x = x * self.imagenet_std.to(x.device) + self.imagenet_mean.to(x.device)
        patches = self.backbone(x).features
        return patches