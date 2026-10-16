import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 通道注意力機制模組：
    賦予模型動態分配256個通道權重的能力，放大關鍵戰術特徵。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out, inplace=True)


class GoRankResNet(nn.Module):
    """
    升級版雙分支入口 + 全局空間金字塔池化 SE-ResNet 架構
    Input: (B, 353, 19, 19)
      - 通道 0 ~ 1    : 全局第 1~250 手「我方 / 對手」落子順序編號平面 (2 通道，輸入 0~250)
      - 通道 2 ~ 51   : 25 個截面 (每 10 步) 的區間內「我方 / 對手」落子順序編號平面 (50 通道，輸入 0~10)
      - 通道 52 ~ 101 : 25 個截面 (每 10 步) 的「我方 / 對手」全盤黑白狀態快照 (50 通道，輸入 0/1)
      - 通道 102 ~ 351: 25 個戰術截面之「我方 / 對手」氣數與眼位死活平面 (250 通道，輸入 0/1)
      - 通道 352      : 全 1 棋盤邊界常數平面 (1 通道)
    Output: (B, 10) - 10 個棋力段位的預測分數
    """
    def __init__(self, in_channels=353, num_channels=256, num_blocks=20, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        half_ch = num_channels // 2  # 128 通道配給歷史與落子順序，128 通道配給戰術死活

        # =====================================================================
        # 【升級 1】雙分支特徵解耦入口 (Dual-Stem Input Decoupling)
        # =====================================================================
        if in_channels == 353:
            # 分支 A：接收 102 個歷史與落子順序通道 (2 全局手數 + 50 區間手數 + 50 截面快照) + 1 個邊界通道 = 103 維
            self.stem_history = nn.Sequential(
                nn.Conv2d(103, half_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(half_ch),
                nn.ReLU(inplace=True)
            )
            # 分支 B：接收 250 個氣數與死活戰術通道 + 1 個全 1 邊界通道 = 251 維
            self.stem_tactical = nn.Sequential(
                nn.Conv2d(251, num_channels - half_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_channels - half_ch),
                nn.ReLU(inplace=True)
            )
        else:
            # 相容其他通道數的通用入口
            self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
            self.bn_in = nn.BatchNorm2d(num_channels)

        # 入口融合層：將雙分支合併後的 256 通道進行特徵互動與注意力校準
        self.stem_fusion = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            SEBlock(num_channels)
        )

        # 主幹網路：20 個 SE-Residual Blocks (維持 256 通道深度特徵提取)
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # =====================================================================
        # 【升級 2】解除尾端 6 通道瓶頸：改用全局平均/最大池化 + 3x3 九宮格區域池化頭
        # 完整保留 256 個通道的戰術資訊，不再將 256 通道強制壓縮為 6 通道
        # =====================================================================
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 3x3 區域池化：將 19x19 棋盤濃縮為「四角、四邊、中腹」9 個戰略區域 (64 * 3 * 3 = 576 維)
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))

        # 總特徵維度 = 全局平均 (256) + 全局最大 (256) + 九宮格區域特徵 (576) = 1088 維
        head_in_features = num_channels * 2 + 64 * 3 * 3

        self.classifier = nn.Sequential(
            nn.Linear(head_in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. 入口雙分支獨立提取與標準化
        if self.in_channels == 353:
            border = x[:, 352:353, :, :]                              # (B, 1, 19, 19) 棋盤邊界平面
            x_global_move = x[:, 0:2, :, :] * (1.0 / 250.0)           # 將全局手數 (0~250) 正規化至 [0, 1]
            x_local_move = x[:, 2:52, :, :] * (1.0 / 10.0)            # 將區間手數 (0~10) 正規化至 [0, 1]
            x_snapshots = x[:, 52:102, :, :]                          # 25 個截面全盤黑白快照 (0/1)
            x_hist = torch.cat([x_global_move, x_local_move, x_snapshots, border], dim=1)  # (B, 103, 19, 19)
            x_tact = torch.cat([x[:, 102:352, :, :], border], dim=1)                       # (B, 251, 19, 19)
            out = torch.cat([self.stem_history(x_hist), self.stem_tactical(x_tact)], dim=1)
        else:
            out = F.relu(self.bn_in(self.conv_in(x)), inplace=True)

        # 2. 入口融合與 20 層 SE-ResNet 主幹推演
        out = self.stem_fusion(out)
        out = self.layers(out)  # (B, 256, 19, 19)

        # 3. 多尺度無瓶頸特徵匯總 (Global Avg + Global Max + 3x3 Spatial Pyramid)
        gap = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)   # (B, 256) 全局平均特徵
        gmp = F.adaptive_max_pool2d(out, 1).view(out.size(0), -1)   # (B, 256) 全局最大峰值特徵
        spatial = self.spatial_pool(self.spatial_conv(out)).view(out.size(0), -1)  # (B, 576) 區域特徵

        feat = torch.cat([gap, gmp, spatial], dim=1)                # (B, 1088)

        # 4. 棋力段位分類輸出
        out = self.classifier(feat)
        return out
