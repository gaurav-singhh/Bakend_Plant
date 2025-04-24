# model_definition.py
import sys
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38, unfreeze_layers=4, dropout_prob=0.3):
        super().__init__()
        # All diagnostic/logging prints go to stderr to avoid polluting stdout
        print("Initializing EfficientNet-B0 model with ImageNet weights", file=sys.stderr)
        self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze all base parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # Unfreeze the last few blocks for fine-tuning
        print(f"Unfreezing the last {unfreeze_layers} blocks", file=sys.stderr)
        for block in self.base.features[-unfreeze_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        # Custom feature projector and classification head
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.base.classifier[1].in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_prob / 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes),
        )

        self._init_weights()
        print("Model initialization complete.", file=sys.stderr)

    def _init_weights(self):
        for layer in [self.feature_projector, self.classifier]:
            for m in layer:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base.features(x)
        x = self.feature_projector(x)
        return self.classifier(x)
