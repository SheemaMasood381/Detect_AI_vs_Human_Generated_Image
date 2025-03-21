import torch
import torch.nn as nn
from timm import create_model


class ViTConXWithAvgPooling(nn.Module):
    def __init__(self, num_classes=1):
        super(ViTConXWithAvgPooling, self).__init__()

        # Load ConvNeXt Large
        self.convnext = create_model("convnext_large", pretrained=False, num_classes=0)
        convnext_out = self.convnext.num_features

        # Load Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
        swin_out = self.swin.num_features

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers for feature fusion
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(convnext_out + swin_out),
            nn.Linear(convnext_out + swin_out, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Decoder: Final Classification Head
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Pass through ConvNeXt and Swin Transformer
        x_convnext = self.convnext(x)
        x_swin = self.swin(x)

        # Apply Global Average Pooling
        x_convnext = self.global_avg_pool(x_convnext.unsqueeze(-1)).squeeze(-1)
        x_swin = self.global_avg_pool(x_swin.unsqueeze(-1)).squeeze(-1)

        # Concatenate both feature vectors
        x_combined = torch.cat((x_convnext, x_swin), dim=1)
        x_fused = self.feature_fusion(x_combined)

        # Final classification
        decoded_output = self.decoder(x_fused)
        return decoded_output
