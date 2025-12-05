"""
Pretrained model loaders
"""
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, c_t=512, act_fn=nn.SiLU, dropout=0.2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, c_t),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(c_t,c_t//2),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(c_t//2,num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x