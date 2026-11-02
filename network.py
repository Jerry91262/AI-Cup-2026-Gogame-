import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力機制：動態強化關鍵手數、新落子與提死子特徵通道。"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x):
        # Squeeze: 全域平均池化濃縮各通道空間響應 (B, C, 19, 19) -> (B, C)
        w = x.mean(dim=(-2, -1))
        # Excitation: 學習各通道重要性權重 (0 ~ 1) 並回乘原特徵圖
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True, se_reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.05)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.05)
        self.se = SEBlock(channels, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class GoRankResNet(nn.Module):
    """
    A ResNet-based model for ranking Go positions.
    Input: (B, 501, 19, 19) -> GPU 即時推導前後新落子 (500) 與提死子 (500)，擴展為 1501 通道
    Output: (B, 10) - scores for the 10 rank classes
    """
    # 擴展模型容量為 256 通道 × 16 層 SE-ResNet 殘差塊，結合全域池化 + 3x3 九宮格區域池化與 256 維分類頭。
    def __init__(self, in_channels=501, num_channels=256, num_blocks=16, num_classes=10):
        super().__init__()
        self.history_length = (in_channels - 1) // 2
        self.history_channels = self.history_length * 2
        # 原始 501 通道 + 逐手新落子 500 通道 + 逐手被提死子 500 通道 = 1501 通道
        total_in_channels = in_channels + self.history_channels * 2

        self.conv_in = nn.Conv2d(total_in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels, momentum=0.05)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels, use_se=True, se_reduction=8) for _ in range(num_blocks)]
        )
        # 3x3 九宮格區域池化：將 19x19 棋盤濃縮為「四角、四邊、中腹」9 個戰略區域 (64 * 3 * 3 = 576 維)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))

        # 總特徵維度 = 全域平均 (256) + 全域最大 (256) + 九宮格區域特徵 (576) = 1088 維
        head_in_features = num_channels * 2 + 64 * 3 * 3
        self.head = nn.Sequential(
            nn.Linear(head_in_features, num_channels),
            nn.BatchNorm1d(num_channels, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(num_channels, num_classes)
        )

    def _append_move_and_capture_planes(self, x):
        """
        於 GPU 即時從前 500 個逐手盤面通道推導「前後新落子」與「被提死子」通道（不增加磁碟分片體積與 I/O 負擔）：
        - new_moves (500 通道): 每一手剛落下的新子位置 (S_t - S_{t-1} == +1)
        - dead_captures (500 通道): 每一手被提掉的死子位置 (S_t - S_{t-1} == -1)
        """
        b = x.size(0)
        boards = x[:, :self.history_channels].reshape(b, self.history_length, 2, 19, 19)

        # 有效手遮罩：真實落子後盤面必有棋子 (> 0)；不足 250 手的末端補零 (Post-padding) 則全盤為 0，
        # 遮罩可防止對局結束切換到補零空盤時，將盤上所有既有棋子誤判為「被提死子」。
        step_valid = (boards[:, 1:].amax(dim=(2, 3, 4), keepdim=True) > 0).to(x.dtype)

        # 逐手差分：第 1 手直接取初始落子（第 1 手不可能提子），第 2~250 手計算前後盤面變化 (S_t - S_{t-1})
        diff_next = (boards[:, 1:] - boards[:, :-1]) * step_valid

        new_moves = torch.cat(
            [boards[:, :1], F.relu(diff_next)], dim=1
        ).reshape(b, self.history_channels, 19, 19)
        dead_captures = torch.cat(
            [torch.zeros_like(boards[:, :1]), F.relu(-diff_next)], dim=1
        ).reshape(b, self.history_channels, 19, 19)

        return torch.cat([x, new_moves, dead_captures], dim=1)

    def forward(self, x):
        x = self._append_move_and_capture_planes(x)
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        avg_pool = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        max_pool = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
        spatial = self.spatial_pool(self.spatial_conv(out)).view(out.size(0), -1)
        out = torch.cat([avg_pool, max_pool, spatial], dim=1)
        out = self.head(out)

        return out

class GoPlayerResNet(nn.Module):
    """
    A ResNet-based model to generate player embeddings.
    Input: (B, 17, 19, 19)
    Output: (B, 128) L2-normalized embedding
    """
    def __init__(self, in_channels=17, num_channels=128, num_blocks=16, embed_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels, use_se=False) for _ in range(num_blocks)]
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Linear(num_channels // 2, embed_dim)
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        out = self.avg_pool(out) # Pooling        
        out = out.view(out.size(0), -1) # Flatten
        out = self.mlp_head(out) # MLP Head
        out = F.normalize(out, p=2, dim=1) # L2 Normalize the embedding (crucial for metric learning)
        return out
