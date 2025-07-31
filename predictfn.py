
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2

# 1. IMPORTY I KONFIGURACJA
import os
import glob
import torch
import numpy as np
import time
import copy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Augmentacje i preprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2

# MONAI dla zaawansowanych funkcji straty
# from monai.losses import DiceLoss as MonaiDice, FocalLoss as MonaiFocal, TverskyLoss as MonaiTversky, DiceCELoss as MonaiDiceCE, DiceFocalLoss as MonaiDiceFocal

# Sprawdzenie dostępności GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'PyTorch używa GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch używa MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("GPU niedostępne, używamy CPU")

print(f"🔧 Device: {device}")
path = '/home/ives/rafal/notebooks/data'







# 4. ARCHITEKTURY MODELI
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ======================= PODSTAWOWE BLOKI =======================

class ResidualBlock(nn.Module):
    """Ulepszona wersja bloku residualnego z opcjonalnym dropout i aktywacją"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Wybór funkcji aktywacji
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.activation(out)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionGate(nn.Module):
    """Attention Gate dla precyzyjnego łączenia feature map"""
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ======================= MODEL 0: Simple ResUNet (Baseline) =======================

class SimpleResUNet(nn.Module):
    """Stabilna wersja oryginalnego ResUNet z poprawionymi wymiarami"""
    def __init__(self, n_class=1, base_channels=64):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, base_channels, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = self._make_layer(base_channels * 8, base_channels * 4, 2)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = self._make_layer(base_channels * 4, base_channels * 2, 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = self._make_layer(base_channels * 2, base_channels, 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = self._make_layer(base_channels * 2, base_channels, 2)
        
        self.final = nn.Conv2d(base_channels, n_class, 1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        conv1 = x
        x = self.maxpool(x)
        
        conv2 = self.layer1(x)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        x = self.layer4(conv4)
        
        # Decoder
        x = self.up4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dec1(x)
        
        return self.final(x)

# ======================= MODEL 1: ResUNet++ (Średni) =======================

class ResUNetPlusPlus(nn.Module):
    """ResUNet++ - ulepszona wersja z nested skip connections"""
    def __init__(self, n_class=1, base_channels=64, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.conv0_0 = self._make_layer(1, base_channels, 2)
        self.conv1_0 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.conv2_0 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.conv3_0 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        self.conv4_0 = self._make_layer(base_channels * 8, base_channels * 16, 2, stride=2)
        
        # Nested skip pathways
        self.conv0_1 = self._make_layer(base_channels + base_channels * 2, base_channels, 2)
        self.conv1_1 = self._make_layer(base_channels * 2 + base_channels * 4, base_channels * 2, 2)
        self.conv2_1 = self._make_layer(base_channels * 4 + base_channels * 8, base_channels * 4, 2)
        self.conv3_1 = self._make_layer(base_channels * 8 + base_channels * 16, base_channels * 8, 2)
        
        self.conv0_2 = self._make_layer(base_channels * 2 + base_channels * 2, base_channels, 2)
        self.conv1_2 = self._make_layer(base_channels * 4 + base_channels * 4, base_channels * 2, 2)
        self.conv2_2 = self._make_layer(base_channels * 8 + base_channels * 8, base_channels * 4, 2)
        
        self.conv0_3 = self._make_layer(base_channels * 3 + base_channels * 2, base_channels, 2)
        self.conv1_3 = self._make_layer(base_channels * 6 + base_channels * 4, base_channels * 2, 2)
        
        self.conv0_4 = self._make_layer(base_channels * 4 + base_channels * 2, base_channels, 2)
        
        # Upsampling layers
        self.up1_0 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(base_channels * 16, base_channels * 16, 2, stride=2)
        
        # Additional upsampling for nested paths
        self.up1_1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)
        
        self.up1_2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        
        self.up1_3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        
        # Final layers
        if deep_supervision:
            self.final1 = nn.Conv2d(base_channels, n_class, 1)
            self.final2 = nn.Conv2d(base_channels, n_class, 1)
            self.final3 = nn.Conv2d(base_channels, n_class, 1)
            self.final4 = nn.Conv2d(base_channels, n_class, 1)
        else:
            self.final = nn.Conv2d(base_channels, n_class, 1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        
        # Nested skip connections - Level 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        
        # Nested skip connections - Level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        
        # Nested skip connections - Level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        
        # Final level
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)

# ======================= MODEL 2: Attention ResUNet (Duży) =======================

class AttentionResUNet(nn.Module):
    """ResUNet z Attention Gates i SE blocks"""
    def __init__(self, n_class=1, base_channels=64, use_se=True, dropout_p=0.1):
        super().__init__()
        self.use_se = use_se
        
        # Encoder - POPRAWIONE WYMIARY
        self.conv1 = nn.Conv2d(1, base_channels, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, base_channels, 3)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 3, stride=2)
        
        # SE blocks
        if use_se:
            self.se1 = SEBlock(base_channels)
            self.se2 = SEBlock(base_channels * 2)
            self.se3 = SEBlock(base_channels * 4)
            self.se4 = SEBlock(base_channels * 8)
        
        # Attention Gates - POPRAWIONE WYMIARY
        self.att4 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.att3 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.att2 = AttentionGate(base_channels, base_channels, base_channels // 2)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)
        
        # Decoder - POPRAWIONE WYMIARY
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = self._make_layer(base_channels * 8, base_channels * 4, 3)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = self._make_layer(base_channels * 4, base_channels * 2, 3)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = self._make_layer(base_channels * 2, base_channels, 3)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = self._make_layer(base_channels * 2, base_channels, 2)
        
        self.final = nn.Conv2d(base_channels, n_class, 1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride, dropout=0.1)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=0.1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        conv1 = x
        x = self.maxpool(x)
        
        conv2 = self.layer1(x)
        if self.use_se: conv2 = self.se1(conv2)
        
        conv3 = self.layer2(conv2)
        if self.use_se: conv3 = self.se2(conv3)
        
        conv4 = self.layer3(conv3)
        if self.use_se: conv4 = self.se3(conv4)
        
        x = self.layer4(conv4)
        if self.use_se: x = self.se4(x)
        
        # Decoder with attention - POPRAWIONE
        x = self.up4(x)
        conv4_att = self.att4(conv4, x)
        x = torch.cat([x, conv4_att], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        conv3_att = self.att3(conv3, x)
        x = torch.cat([x, conv3_att], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        conv2_att = self.att2(conv2, x)
        x = torch.cat([x, conv2_att], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        conv1_att = self.att1(conv1, x)
        x = torch.cat([x, conv1_att], dim=1)
        x = self.dec1(x)
        
        return self.final(x)

# ======================= MODEL 3: Deep ResUNet (Bardzo duży) =======================

class DeepResUNet(nn.Module):
    """Bardzo głęboki ResUNet inspirowany ResNet-152"""
    def __init__(self, n_class=1, base_channels=64, layers=[3, 8, 36, 3]):
        super().__init__()
        
        # Initial conv - BEZ stride=2, żeby zachować wymiary
        self.conv1 = nn.Conv2d(1, base_channels, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder layers
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0])
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, layers[3], stride=2)
        
        # Bridge with dilated convolutions
        self.bridge = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, 3, padding=4, dilation=4),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 8, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with more layers
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = self._make_layer(base_channels * 8, base_channels * 4, 6)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = self._make_layer(base_channels * 4, base_channels * 2, 8)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = self._make_layer(base_channels * 2, base_channels, 4)
        
        # POPRAWIONE - upsampling do oryginalnego rozmiaru
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = self._make_layer(base_channels * 2, base_channels, 3)
        
        # Final conv bez dodatkowego upsamplingu
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, n_class, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride, dropout=0.1, activation='leaky_relu')]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=0.1, activation='leaky_relu'))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder - zapisujemy feature mapy dla skip connections
        original_size = x.shape[2:]  # Zapisz oryginalny rozmiar
        
        x = F.relu(self.bn1(self.conv1(x)))
        conv0 = x  # 512x512
        x = self.maxpool(x)
        
        conv1 = self.layer1(x)  # 256x256
        conv2 = self.layer2(conv1)  # 128x128  
        conv3 = self.layer3(conv2)  # 64x64
        conv4 = self.layer4(conv3)  # 32x32
        
        # Bridge
        x = self.bridge(conv4)  # 32x32
        
        # Decoder z prawidłowymi wymiarami
        x = self.up4(x)  # 64x64
        x = torch.cat([x, conv3], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)  # 128x128
        x = torch.cat([x, conv2], dim=1) 
        x = self.dec3(x)
        
        x = self.up2(x)  # 256x256
        x = torch.cat([x, conv1], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)  # 512x512
        x = torch.cat([x, conv0], dim=1)
        x = self.dec1(x)
        
        return self.final_conv(x)

# ======================= FUNKCJE POMOCNICZE =======================

def count_parameters(model):
    """Liczy parametry modelu"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model, input_size=(1, 1, 512, 512)):
    """Zwraca informacje o modelu"""
    total_params = count_parameters(model)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(input_size)
        try:
            output = model(x)
            if isinstance(output, list):
                output_shape = [o.shape for o in output]
            else:
                output_shape = output.shape
        except Exception as e:
            output_shape = f"Error: {e}"
    
    return {
        'parameters': total_params,
        'parameters_M': total_params / 1e6,
        'output_shape': output_shape
    }


def double_conv(in_channels, out_channels):
    """Standardowy blok podwójnej konwolucji"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def enhanced_double_conv(in_channels, out_channels, dropout_p=0.1):
    """Rozszerzony blok z batch normalization i dropout"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout_p),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout_p)
    )

# Original UNet
class UNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        # Encoder
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

# Enhanced UNet z większą liczbą parametrów
class EnhancedUNet(nn.Module):
    def __init__(self, n_class=1, dropout_p=0.1, base_channels=64):
        super().__init__()

        # Encoder (5 poziomów)
        self.dconv_down1 = enhanced_double_conv(1, base_channels, dropout_p)
        self.dconv_down2 = enhanced_double_conv(base_channels, base_channels * 2, dropout_p)
        self.dconv_down3 = enhanced_double_conv(base_channels * 2, base_channels * 4, dropout_p)
        self.dconv_down4 = enhanced_double_conv(base_channels * 4, base_channels * 8, dropout_p)
        self.dconv_down5 = enhanced_double_conv(base_channels * 8, base_channels * 16, dropout_p)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.dconv_up4 = enhanced_double_conv(base_channels * 8 + base_channels * 16, base_channels * 8, dropout_p)
        self.dconv_up3 = enhanced_double_conv(base_channels * 4 + base_channels * 8, base_channels * 4, dropout_p)
        self.dconv_up2 = enhanced_double_conv(base_channels * 2 + base_channels * 4, base_channels * 2, dropout_p)
        self.dconv_up1 = enhanced_double_conv(base_channels + base_channels * 2, base_channels, dropout_p)

        self.conv_last = nn.Conv2d(base_channels, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

# Residual blocks dla ResUNet
class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResUNet(nn.Module):
    def __init__(self, n_class=1, base_channels=64):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, base_channels, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = self._make_layer(base_channels * 8, base_channels * 4, 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = self._make_layer(base_channels * 4, base_channels * 2, 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = self._make_layer(base_channels * 2, base_channels, 2)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = self._make_layer(base_channels * 2, base_channels, 2)

        self.final = nn.Conv2d(base_channels, n_class, 1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock2(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        conv1 = x
        x = self.maxpool(x)

        conv2 = self.layer1(x)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        x = self.layer4(conv4)

        # Decoder
        x = self.up4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dec1(x)

        return self.final(x)

# Funkcja do tworzenia modeli
def create_model(model_type='unet', n_class=1):
    """Factory function dla tworzenia modeli"""
    if model_type == 'unet':
        return UNet(n_class)
    elif model_type == 'enhanced_unet':
        return EnhancedUNet(n_class, dropout_p=0.1, base_channels=64)
    elif model_type == 'resunet' or model_type == 'ResUNet':
        return ResUNet(n_class, base_channels=64)
    elif model_type == 'resunetpp':
        return ResUNetPlusPlus(n_class)
    elif model_type == 'attention_resunet':
        return AttentionResUNet(n_class, dropout_p=0.1, base_channels=64)
    elif model_type == 'deep_resunet':
        return DeepResUNet(n_class, base_channels=64)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

# Porównanie liczby parametrów
def compare_model_sizes():
    """Porównanie rozmiarów modeli"""
    models = {
        'UNet': UNet(1),
        'Enhanced UNet': EnhancedUNet(1),
        'ResUNet': ResUNet(1),
        'ResUNetPP': ResUNetPlusPlus(1),
        'AttentionResUNet': AttentionResUNet(1),
        'DeepResUNet': DeepResUNet(1)
    }

    print("📊 PORÓWNANIE MODELI:")
    print("-" * 40)

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:15}: {total_params:,} parametrów ({trainable_params:,} trenowalnych)")

    return models

# Porównaj modele
# model_comparison = compare_model_sizes()

# 5. PREPROCESSING I DATASET

class ImagePreprocessor:
    """Preprocessing obrazów medycznych"""
    def __init__(self, apply_clahe=True, apply_denoise=True, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
        self.apply_clahe = apply_clahe
        self.apply_denoise = apply_denoise
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # CLAHE dla poprawy kontrastu
        if self.apply_clahe:
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
            image_np = clahe.apply(image_np)

        # Denoising
        if self.apply_denoise:
            if len(image_np.shape) == 2:
                image_np = cv2.fastNlMeansDenoising(image_np, None, 10, 7, 21)
            else:
                image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 10, 10, 7, 21)

        return Image.fromarray(image_np)

class CoronaryDataset(Dataset):
    """Dataset dla nowej struktury folderów data/imgs i data/masks"""

    def __init__(self, data_path=path,
                 train=True, train_split=0.8, apply_preprocessing=True, apply_augmentations=True):

        self.data_path = data_path
        self.imgs_path = os.path.join(data_path, "imgs")
        self.masks_path = os.path.join(data_path, "masks")
        self.train = train
        self.apply_preprocessing = apply_preprocessing
        self.apply_augmentations = apply_augmentations

        # Znajdź wszystkie pliki
        img_files = sorted(glob.glob(os.path.join(self.imgs_path, "*.jpg")))
        mask_files = sorted(glob.glob(os.path.join(self.masks_path, "*.png")))

        # Sparuj pliki po nazwach
        img_names = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        mask_names = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

        # Znajdź wspólne nazwy
        common_names = sorted(list(set(img_names.keys()).intersection(set(mask_names.keys()))))

        # Podziel na train/val
        split_idx = int(len(common_names) * train_split)

        if train:
            self.names = common_names[:split_idx]
        else:
            self.names = common_names[split_idx:]

        self.img_paths = [img_names[name] for name in self.names]
        self.mask_paths = [mask_names[name] for name in self.names]

        print(f"{'🏋️ Train' if train else '🔍 Validation'} dataset: {len(self.names)} próbek")

        # Preprocessor
        self.preprocessor = ImagePreprocessor() if apply_preprocessing else None

        # Augmentacje dostosowane do grayscale (jeśli obrazy są grayscale)
        if train and apply_augmentations:
            # Augmentacje geometryczne (bezpieczne dla grayscale)
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Removed alpha_affine
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(distort_limit=0.05, p=0.2),  # Removed shift_limit
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(p=0.2),  # Changed to var_limit range
                A.Blur(blur_limit=5, p=0.2),
                A.Resize(384, 384),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            # Tylko podstawowe transformacje dla walidacji
            self.augment = A.Compose([
                # A.Resize(512, 512),
                # A.Resize(448, 448),
                A.Resize(384, 384),
                # A.Resize(256, 256),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def __getitem__(self, index):
        # Załaduj obraz i maskę
        image = Image.open(self.img_paths[index])
        mask = Image.open(self.mask_paths[index])

        # Konwertuj na grayscale jeśli potrzeba
        if image.mode != 'L':
            image = image.convert('L')
        if mask.mode != 'L':
            mask = mask.convert('L')

        # Preprocessing
        if self.preprocessor:
            image = self.preprocessor(image)

        # Konwertuj do numpy dla albumentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Zastosuj augmentacje
        augmented = self.augment(image=image_np, mask=mask_np)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']

        # Zapewnij właściwe wymiary
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 3:
            # Jeśli maska ma 3 kanały, weź pierwszy
            mask_tensor = mask_tensor[0:1]

        # Binaryzuj maskę
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.names)

# Funkcja do tworzenia dataloaderów
def create_dataloaders(data_path=path,
                      batch_size=16, train_split=0.8,
                      apply_preprocessing=True, apply_augmentations=True, num_workers=0):
    """Stwórz train i validation dataloaders"""

    train_dataset = CoronaryDataset(
        data_path=data_path,
        train=True,
        train_split=train_split,
        apply_preprocessing=apply_preprocessing,
        apply_augmentations=apply_augmentations
    )

    val_dataset = CoronaryDataset(
        data_path=data_path,
        train=False,
        train_split=train_split,
        apply_preprocessing=apply_preprocessing,
        apply_augmentations=False  # Bez augmentacji dla walidacji
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataloaders utworzone:")
    print(f"  - Train: {len(train_dataset)} próbek, {len(train_loader)} batchy")
    print(f"  - Validation: {len(val_dataset)} próbek, {len(val_loader)} batchy")

    return {'train': train_loader, 'val': val_loader}




def remove_double_frames(image):
    """
    Usuwa podwójne ramki z obrazu medycznego
    
    Args:
        image: PIL Image lub numpy array
    
    Returns:
        image: Obraz z usuniętymi ramkami
    """
    import cv2
    import numpy as np
    
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image)
    
    # Konwertuj do grayscale jeśli potrzebne
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Wykryj krawędzie
    edges = cv2.Canny(gray, 50, 150)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtruj kontury - usuń te które wyglądają jak ramki
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimalny rozmiar
            # Sprawdź czy to prostokąt (ramka)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Jeśli to prostokąt o rozsądnych proporcjach, prawdopodobnie to ramka
            if 0.5 < aspect_ratio < 2.0 and area > 5000:
                continue  # Pomiń ramki
            filtered_contours.append(contour)
    
    # Stwórz maskę bez ramek
    mask = np.ones_like(gray) * 255
    cv2.fillPoly(mask, filtered_contours, 0)
    
    # Zastosuj maskę
    result = cv2.bitwise_and(img_array, img_array, mask=mask)
    
    return result

def preprocess_single_image(image_path, resolution=512, apply_preprocessing=True, remove_frames=False):
    """
    Preprocessing pojedynczego obrazu identyczny jak w treningu

    Args:
        image_path: Ścieżka do obrazu
        resolution: Rozdzielczość modelu (256, 384, 512, etc.)
        apply_preprocessing: Czy zastosować CLAHE i denoising
        remove_frames: Czy usunąć podwójne ramki

    Returns:
        tensor: Przetworzony tensor gotowy do inferencji
    """

    # Załaduj obraz
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path  # Już załadowany PIL Image

    # Usuń podwójne ramki jeśli wymagane
    if remove_frames:
        print("🔧 Removing double frames from image...")
        image = remove_double_frames(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

    # Konwertuj na grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Preprocessing (CLAHE + denoising) - identyczny jak w treningu
    if apply_preprocessing:
        preprocessor = ImagePreprocessor(
            apply_clahe=True,
            apply_denoise=True,
            clahe_clip_limit=3.0,
            clahe_tile_grid_size=(8, 8)
        )
        image = preprocessor(image)

    # Konwertuj do numpy
    image_np = np.array(image)

    # Augmentacje (tylko resize i normalizacja - jak w walidacji)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(resolution, resolution),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    # Zastosuj transformacje
    transformed = transform(image=image_np)
    image_tensor = transformed['image']

    # Dodaj batch dimension
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Dodaj channel dim

    image_tensor = image_tensor.unsqueeze(0)  # Dodaj batch dim: (1, 1, H, W)

    return image_tensor, image_np

def load_trained_model_for_inference(model_path, model_type=None, device=None):
    """
    Załaduj wytrenowany model do inferencji

    Args:
        model_path: Ścieżka do zapisanego modelu (.pth)
        model_type: Typ modelu ('unet', 'enhanced_unet', 'resunet')
        device: Urządzenie ('cuda', 'mps', 'cpu', lub None=auto)

    Returns:
        model: Załadowany model w trybie eval
        device: Urządzenie na którym jest model
    """

    # Auto-detect device
    if device is None:
        print(f"Auto-detecting device...")
        print(f"Environment variables:")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS available, using MPS")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU")

    print(f"Loading model on: {device}")

    # Stwórz model
    model = create_model(model_type, n_class=1)

    # Załaduj wagi
    try:
        checkpoint = torch.load(model_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Training loss: {checkpoint.get('loss', 'unknown')}")
        else:
            # Stary format - tylko state_dict
            model.load_state_dict(checkpoint)
            print(f"Model loaded (legacy format)")

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, device

    # Przenieś na device i ustaw eval mode
    model = model.to(device)
    model.eval()

    return model, device

def inference_single_image(model, image_tensor, device, threshold=0.5):
    """
    Inferencja pojedynczego obrazu

    Args:
        model: Załadowany model
        image_tensor: Przetworzony tensor obrazu
        device: Urządzenie
        threshold: Próg binaryzacji (0.5)

    Returns:
        prediction: Predykcja jako tensor
        prediction_binary: Zbinaryzowana predykcja
        confidence: Średnia pewność predykcji
    """

    with torch.no_grad():
        # Przenieś na device
        print(f"Moving input tensor to device: {device}")
        image_tensor = image_tensor.to(device)
        print(f"Input tensor device: {image_tensor.device}")
        print(f"Model device: {next(model.parameters()).device}")

        # Forward pass
        print(f"Running inference on device: {device}")
        output = model(image_tensor)

        # Sigmoid dla prawdopodobieństw
        prediction = torch.sigmoid(output)

        # Binaryzacja
        prediction_binary = (prediction > threshold).float()

        # Statystyki
        confidence = prediction.mean().item()
        positive_pixels = prediction_binary.sum().item()
        total_pixels = prediction_binary.numel()
        positive_ratio = positive_pixels / total_pixels

        print(f"Inference stats:")
        print(f"  - Average confidence: {confidence:.4f}")
        print(f"  - Positive pixels: {positive_pixels}/{total_pixels} ({positive_ratio*100:.2f}%)")

        return prediction, prediction_binary, confidence

def remove_small_components_from_mask(mask, min_size=500):
    """
    Usuwa małe komponenty z binarnej maski (np. < 500 pikseli)

    Args:
        mask: Binarna maska jako numpy array (0 i 1)
        min_size: Minimalny rozmiar komponentu do pozostawienia (w pikselach)

    Returns:
        mask_cleaned: Oczyszczona maska
    """
    import cv2
    import numpy as np

    # Konwertuj na uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Znajdź komponenty połączone
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    # Stwórz nową pustą maskę
    mask_cleaned = np.zeros_like(mask, dtype=np.uint8)

    # Zostaw tylko duże komponenty
    for i in range(1, num_labels):  # pomiń tło (i=0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask_cleaned[labels == i] = 1

    return mask_cleaned


def visualize_inference_result(original_image, prediction, prediction_binary,
                              save_path=None, title="Coronary Artery Segmentation"):
    """
    Wizualizacja wyników inferencji

    Args:
        original_image: Oryginalny obraz (numpy array)
        prediction: Soft prediction (tensor)
        prediction_binary: Binary prediction (tensor)
        save_path: Ścieżka do zapisania (opcjonalne)
        title: Tytuł wykresu
    """

    # Konwertuj tensory na numpy
    pred_soft = prediction.cpu().squeeze().numpy()
    pred_binary = prediction_binary.cpu().squeeze().numpy()

    # Stwórz subplot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Oryginalny obraz
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 2. Soft prediction (heatmapa)
    im1 = axes[1].imshow(pred_soft, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Prediction Heatmap\\nMax: {pred_soft.max():.3f}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    # 3. Binary prediction
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Binary Segmentation')
    axes[2].axis('off')

    # 4. Overlay
    # Resize original to match prediction
    if original_image.shape != pred_binary.shape:
        original_resized = cv2.resize(original_image, (pred_binary.shape[1], pred_binary.shape[0]))
    else:
        original_resized = original_image

    # Stwórz overlay
    overlay = original_resized.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB) if len(overlay.shape) == 2 else overlay

    # Dodaj maskę w kolorze czerwonym
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = pred_binary * 255  # Czerwony kanał

    # Blend
    overlay_blended = cv2.addWeighted(overlay, 0.7, red_mask.astype(np.uint8), 0.3, 0)

    axes[3].imshow(overlay_blended)
    axes[3].set_title('Overlay (Red = Arteries)')
    axes[3].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Zapisz jeśli podano ścieżkę
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Result saved to: {save_path}")

    plt.show()

    # Zwróć statystyki
    stats = {
        'max_confidence': pred_soft.max(),
        'mean_confidence': pred_soft.mean(),
        'positive_pixels': pred_binary.sum(),
        'positive_ratio': pred_binary.mean(),
        'image_shape': original_image.shape,
        'prediction_shape': pred_binary.shape
    }

    return stats

def full_inference_pipeline(image_path, model_path, model_type=None,
                           resolution=512, threshold=0.5, save_result='/home/ives/rafal/notebooks2/notebooks/'):
    """
    Kompletny pipeline inferencji pojedynczego obrazu

    Args:
        image_path: Ścieżka do obrazu
        model_path: Ścieżka do modelu
        model_type: Typ modelu
        resolution: Rozdzielczość
        threshold: Próg binaryzacji
        save_result: Ścieżka do zapisania wyniku

    Returns:
        results: Słownik z wynikami
    """

    print(f"FULL INFERENCE PIPELINE")
    print(f"="*50)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Threshold: {threshold}")

    try:
        # 1. Załaduj model
        print(f"\\nLoading model...")
        model, device = load_trained_model_for_inference(model_path, model_type)
        if model is None:
            return None

        # 2. Preprocessing obrazu
        print(f"\\n2Preprocessing image...")
        image_tensor, original_image = preprocess_single_image(image_path, resolution)
        print(f"   Original shape: {original_image.shape}")
        print(f"   Tensor shape: {image_tensor.shape}")

        # 3. Inferencja
        print(f"\\n3Running inference...")
        prediction, prediction_binary, confidence = inference_single_image(
            model, image_tensor, device, threshold
        )

        # 3.5 Usuwanie małych komponentów
        print(f"\nPostprocessing (remove small objects)...")
        prediction_binary_np = prediction_binary.cpu().squeeze().numpy()
        prediction_binary_clean = remove_small_components_from_mask(prediction_binary_np, min_size=300)

        # Konwertuj z powrotem do tensora
        prediction_binary = torch.tensor(prediction_binary_clean).unsqueeze(0).unsqueeze(0).to(device).float()

        # 4. Wizualizacja
        print(f"\\n4Visualizing results...")
        stats = visualize_inference_result(
            original_image, prediction, prediction_binary,
            save_path=save_result,
            title=f"Coronary Segmentation - {Path(image_path).name}"
        )

        # 5. Podsumowanie
        print(f"\\nINFERENCE COMPLETED!")
        print(f"Results:")
        print(f"   Max confidence: {stats['max_confidence']:.4f}")
        print(f"   Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"   Positive pixels: {stats['positive_pixels']:.0f} ({stats['positive_ratio']*100:.2f}%)")

        return {
            'model': model,
            'device': device,
            'original_image': original_image,
            'prediction': prediction,
            'prediction_binary': prediction_binary,
            'stats': stats,
            'image_path': image_path,
            'model_path': model_path
        }

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

print("FUNKCJE INFERENCJI GOTOWE!")
print("\\nGłówna funkcja:")
print("results = full_inference_pipeline(image_path, model_path)")
print("\\nDostępne funkcje:")
print("- preprocess_single_image() - preprocessing")
print("- load_trained_model_for_inference() - ładowanie modelu")
print("- inference_single_image() - inferencja")
print("- visualize_inference_result() - wizualizacja")
print("- full_inference_pipeline() - kompletny pipeline")
def batch_inference_folder(folder_path, model_path, model_type=None,
                          resolution=512, save_folder=None):
    """
    Batch inferencja całego folderu obrazów

    Args:
        folder_path: Ścieżka do folderu z obrazami
        model_path: Ścieżka do modelu
        model_type: Typ modelu
        resolution: Rozdzielczość
        save_folder: Folder do zapisania wyników
    """

    print(f"📁 BATCH INFERENCE - FOLDER PROCESSING")
    print(f"="*60)
    print(f"📂 Source folder: {folder_path}")
    print(f"🧠 Model: {model_path}")

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return None

    # Znajdź wszystkie obrazy
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return None

    print(f"📊 Found {len(image_files)} images")

    # Stwórz folder wyników
    if save_folder is None:
        save_folder = f"{folder_path}_results_{resolution}"

    os.makedirs(save_folder, exist_ok=True)
    print(f"💾 Results will be saved to: {save_folder}")

    # Załaduj model raz
    print(f"\\nLoading model...")
    model, device = load_trained_model_for_inference(model_path, model_type)
    if model is None:
        return None

    # Batch processing
    results = []
    successful = 0

    print(f"\\nStarting batch processing...")

    for i, image_file in enumerate(image_files):
        print(f"\\n📷 [{i+1}/{len(image_files)}] Processing: {image_file.name}")

        try:
            # Preprocessing
            image_tensor, original_image = preprocess_single_image(str(image_file), resolution)

            # Inferencja
            prediction, prediction_binary, confidence = inference_single_image(
                model, image_tensor, device, threshold=0.5
            )

            print(f"\nPostprocessing (remove small objects)...")
            prediction_binary_np = prediction_binary.cpu().squeeze().numpy()
            prediction_binary_clean = remove_small_components_from_mask(prediction_binary_np, min_size=300)

            # Konwertuj z powrotem do tensora
            prediction_binary = torch.tensor(prediction_binary_clean).unsqueeze(0).unsqueeze(0).to(device).float()

            # Zapisz wyniki
            result_path = os.path.join(save_folder, f"result_{image_file.stem}.png")
            stats = visualize_inference_result(
                original_image, prediction, prediction_binary,
                save_path=result_path,
                title=f"Coronary Segmentation - {image_file.name}"
            )

            results.append({
                'filename': image_file.name,
                'stats': stats,
                'result_path': result_path
            })

            successful += 1
            print(f"Success - Confidence: {stats['mean_confidence']:.4f}")

        except Exception as e:
            print(f"❌ Failed: {e}")

    # Podsumowanie
    print(f"\\nBATCH PROCESSING COMPLETED!")
    print(f"Successful: {successful}/{len(image_files)}")
    print(f"Results saved in: {save_folder}")

    # Stwórz podsumowanie CSV
    try:
        import pandas as pd

        summary_data = []
        for result in results:
            summary_data.append({
                'filename': result['filename'],
                'max_confidence': result['stats']['max_confidence'],
                'mean_confidence': result['stats']['mean_confidence'],
                'positive_pixels': result['stats']['positive_pixels'],
                'positive_ratio_percent': result['stats']['positive_ratio'] * 100,
                'result_path': result['result_path']
            })

        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(save_folder, 'batch_results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Summary saved: {csv_path}")

        # Pokaż statystyki
        print(f"\\nBATCH STATISTICS:")
        print(f"Mean confidence: {df['mean_confidence'].mean():.4f} ± {df['mean_confidence'].std():.4f}")
        print(f"Mean positive ratio: {df['positive_ratio_percent'].mean():.2f}% ± {df['positive_ratio_percent'].std():.2f}%")

    except ImportError:
        print("Install pandas for CSV summary: !pip install pandas")

    return results


# Pojedynczy obraz - kompletny pipeline
# results = full_inference_pipeline("/home/ives/rafal/notebooks/wum.png", "/home/ives/rafal/notebooks2/notebooks/best_multiscale_model_4.pth", model_type='resunet', resolution=256)

# Upload w Colab
# results = colab_upload_and_inference("/content/drive/MyDrive/Colab Notebooks/best_resunet_monai_dice.pth")

# Batch folder
# results = batch_inference_folder("/home/ives/rafal/notebooks2/notebooks/test", "/home/ives/rafal/notebooks2/notebooks/best_attention_resunet_dice_16_50.pth", model_type='attention_resunet', resolution=320)

