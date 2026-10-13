import torch
import torch.nn as nn
import torch.nn.functional as F

# 【新增 1】Squeeze-and-Excitation (SE) 注意力機制模組
class SEBlock(nn.Module):
    """
    賦予模型「抓重點」的能力，自動放大重要特徵，抑制無用特徵。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 步驟 A: Squeeze (擠壓) - 將 19x19 的空間資訊壓縮成 1x1 的全局特徵，讓大腦掌握整體大局
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 步驟 B: Excitation (激發) - 透過兩層網路學習每個通道的「重要性權重」
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # 將權重壓縮到 0~1 之間
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 1. 執行 Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # 2. 執行 Excitation，得到各通道的權重
        y = self.fc(y).view(b, c, 1, 1)
        
        # 3. 將計算出的權重乘回原本的特徵圖上 (自動分配注意力)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 【新增 2】在原有的殘差塊中，宣告我們剛剛建立的 SE 雷達
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 【新增 3】在資料與捷徑 (residual) 合併之前，先讓 SE 模組進行特徵的「權重重新分配」
        out = self.se(out)
        
        out += residual
        return F.relu(out)


class GoRankResNet(nn.Module):
    """
    A ResNet-based model for ranking Go positions (現在升級為 SE-ResNet 架構).
    Input: (B, 500, 19, 19)
    Output: (B, 10) - scores for the 10 rank classes
    """
    def __init__(self, in_channels=500, num_channels=256, num_blocks=20, num_classes=10):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.conv_in2 = nn.Conv2d(num_channels, 6, kernel_size=1)
        self.bn_in2 = nn.BatchNorm2d(6)
        self.fc = nn.Linear(6 * 19 * 19, 2 * 19 * 19)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(2 * 19 * 19, num_classes)

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks (內部已包含 SE 注意力機制)
        out = F.relu(self.bn_in2(self.conv_in2(out))) 
        out = out.view(out.size(0), -1) # Flatten
        out = F.relu(self.fc(out))
        out = self.dropout(out)  # Apply dropout before final layer
        out = self.fc_2(out)

        return out
