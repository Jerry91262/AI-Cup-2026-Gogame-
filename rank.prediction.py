import sys
import copy
import glob
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import webdataset as wds
from torch.utils.data import DataLoader

try:
    from sklearn.model_selection import train_test_split
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from utils import SGFParseRankPrediction
from network import GoRankResNet
'''
10/30 進階強化版本（逐手新落子 & 提死子動態特徵 + SE-ResNet + Model EMA）
1. 核心架構與訓練配置：
   - utils.py: 磁碟分片維持前 250 手逐手「目標玩家 / 對手」二元盤面 (500 通道) + 執黑/執白通道 (1 通道) = 501 通道（零額外硬碟與 I/O 負擔）
   - network.py: 於 GPU forward 即時推導逐手「新落子 (500 通道)」與「被提死子 (500 通道)」擴展為 1501 通道；
                 主幹升級為 256 通道 × 16 層 SE-ResNet (Squeeze-and-Excitation, BN momentum=0.05) + 1088 維多尺度空間池化頭 (參數量 22,923,146)
   - rank-prediction.py: 加入 Model EMA (decay=0.9995 + 動態暖身) 平滑權重與 BN 統計量，驗證與最佳權重儲存均採用 EMA 模型
   - 損失與排程: OrdinalLabelSmoothingLoss(smoothing=0.05) + AdamW(lr=1e-3, weight_decay=1e-2) + 2 輪 Warmup + CosineAnnealingLR + 8 方向 TTA
'''
# --- 加入日誌記錄功能 ---
class Logger(object):
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8') # 每次執行會覆寫，若要附加請改 "a"

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 將標準輸出重導向到 Logger
sys.stdout = Logger("output_log.txt")
# ------------------------

print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| webdataset', wds.__version__)


# 官方資料放置位置：training/ 與 tests/ 依官方發佈的結構擺放
TRAIN_DIR = './training'
TEST_DIR = './tests'
FEAT_DIR = './rank-features'
CKPT_DIR = './rank-trained-models'

# 等級由弱到強。順序很重要：評分時的「相鄰一級」就是這個順序上的相鄰。
LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK_TO_ID = {lv: i for i, lv in enumerate(LEVELS)}
ID_TO_RANK = {i: lv for lv, i in RANK_TO_ID.items()}

NPROC = 8              # 特徵抽取的進程數，設成 CPU 核數
PER_SHARD = 1000       # 每個 webdataset 分片的樣本數
SEED = 42
VAL_SIZE = 100_000     # 從訓練集切出來的驗證集大小

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 2
NUM_EPOCHS = 30
EMA_DECAY = 0.9995
NUM_CLASSES = len(LEVELS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(CKPT_DIR, exist_ok=True)
print('device', DEVICE)
RANK_TO_ID


df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{TRAIN_DIR}/train_*.csv'))],
               ignore_index=True)
if _HAS_SKLEARN:
    df_train, df_val = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=df['rank']
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
else:
    # 若環境未安裝 scikit-learn，使用等效的固定隨機種子分層抽樣 (每段位精確切出 10,000 局驗證集)
    per_rank_val = VAL_SIZE // NUM_CLASSES
    val_parts, train_parts = [], []
    for lv in sorted(df['rank'].unique()):
        sub = df[df['rank'] == lv].sample(frac=1.0, random_state=SEED)
        val_parts.append(sub.iloc[:per_rank_val])
        train_parts.append(sub.iloc[per_rank_val:])
    df_val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print('整體比例')
print(df['rank'].value_counts(normalize=True).sort_index())

print('訓練集比例')
print(df_train['rank'].value_counts(normalize=True).sort_index())

print('驗證集比例')
print(df_val['rank'].value_counts(normalize=True).sort_index())
df.head(3)


_parser = SGFParseRankPrediction()
_demo = df.iloc[0]['sgf_content']
_feat, _err = _parser.features_from_sgf_content(_demo, df.iloc[0]['color'])
print('一局棋的特徵形狀:', _feat.shape, _feat.dtype)


def extract_worker(df_part, pattern, wid):
    """抽一份切片的特徵並寫成分片。每個進程獨立持有 parser 與 ShardWriter。"""
    parser = SGFParseRankPrediction()
    written = bad = 0
    t0 = time.time()
    with wds.ShardWriter(pattern, maxcount=PER_SHARD, verbose=0) as sink:
        for _, row in df_part.iterrows():
            feat, err = parser.features_from_sgf_content(row['sgf_content'], row['color'])
            if err is not None:
                bad += 1
                continue
            sink.write({
                '__key__': '%s_%s' % (row['rank'], row['game_id']),
                'features.npy': feat,
                'label.txt': str(row['rank']),
                'rank_idx.txt': str(RANK_TO_ID[row['rank']]),
                'game_name.txt': str(row['game_id']),
            })
            written += 1
            if written % 20000 == 0:
                print('  [w%d] %d/%d  %.0f 分'
                      % (wid, written, len(df_part), (time.time() - t0) / 60),
                      flush=True)
    print('  [w%d] 寫入 %d 筆（解析失敗 %d）%.1f 分'
          % (wid, written, bad, (time.time() - t0) / 60), flush=True)


def extract_split(df_part, tag):
    """跑完一個 split，並確認產出真的有內容。"""
    outdir = f'{FEAT_DIR}/{tag}'
    os.makedirs(outdir, exist_ok=True)
    for old_f in glob.glob(f'{outdir}/*.tar.gz'):
        os.remove(old_f)

    procs = []
    for i in range(NPROC):
        pat = '%s/%s-w%d-%%06d.tar.gz' % (outdir, tag, i)
        p = Process(target=extract_worker, args=(df_part.iloc[i::NPROC], pat, i))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit('%s: worker 非正常結束（exitcode %s）' % (tag, p.exitcode))

    shards = glob.glob(f'{outdir}/*.tar.gz')
    total = sum(os.path.getsize(s) for s in shards)
    avg = total / len(shards) if shards else 0
    print('%s: %d 個分片, %.2f GB, 平均 %.1f KB/分片'
          % (tag, len(shards), total / 1e9, avg / 1024), flush=True)
    # 平均分片大小是最簡單的健全性檢查：寫入迴圈若出錯，計數器仍會累加、log 看起來
    # 正常，但 tar 是空的（gzip 後僅 46 bytes）。只數分片數量抓不到這種情況。
    if avg < 10 * 1024:
        raise SystemExit('%s: 分片平均僅 %.0f bytes，判定為空分片' % (tag, avg))


def shards_match_expected_shape():
    train_s = sorted(glob.glob(f'{FEAT_DIR}/train/*.tar.gz'))
    val_s = sorted(glob.glob(f'{FEAT_DIR}/val/*.tar.gz'))
    if not train_s or not val_s:
        return False
    try:
        sample = next(iter(wds.WebDataset([val_s[0]], shardshuffle=False).decode()))
        val0_feat, _ = _parser.features_from_sgf_content(df_val.iloc[0]['sgf_content'], df_val.iloc[0]['color'])
        return sample['features.npy'].shape == val0_feat.shape and np.array_equal(sample['features.npy'], val0_feat)
    except Exception:
        return False


# 一次性作業。分片產出且特徵形狀相符時可直接跳過；若特徵通道數改變則自動重抽分片。
if shards_match_expected_shape():
    print('偵測到分片已存在且形狀相符 %s，跳過特徵抽取！' % (str(_feat.shape),))
else:
    print('開始抽取特徵分片（目標形狀 %s）...' % (str(_feat.shape),), flush=True)
    t0 = time.time()
    extract_split(df_val, 'val')
    extract_split(df_train, 'train')
    print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))


def decode_sample(sample):
    return (torch.from_numpy(sample['features.npy']).float(),
            torch.tensor(int(sample['rank_idx.txt']), dtype=torch.long),
            sample['game_name.txt'])


def make_loader(shards, workers, shuffle):
    ds = wds.WebDataset(shards, shardshuffle=1000 if shuffle else 0, detshuffle=shuffle)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.decode().map(decode_sample).batched(BATCH_SIZE, partial=True)
    return DataLoader(ds, batch_size=None, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


def pick_workers(val_shards, candidate=4):
    """數一遍驗證集：筆數對得上才用多 worker，否則退回單執行緒。"""
    total = sum(len(y) for _x, y, _n in make_loader(val_shards, candidate, False))
    if total == VAL_SIZE:
        print('num_workers=%d 驗證通過（%d 筆）' % (candidate, total))
        return candidate
    print('num_workers=%d 讀出 %d 筆（應為 %d），退回 0' % (candidate, total, VAL_SIZE))
    return 0


train_shards = sorted(glob.glob(f'{FEAT_DIR}/train/*.tar.gz'))
val_shards = sorted(glob.glob(f'{FEAT_DIR}/val/*.tar.gz'))
print('分片：train %d / val %d' % (len(train_shards), len(val_shards)))

workers = pick_workers(val_shards)
train_loader = make_loader(train_shards, workers, True)
val_loader = make_loader(val_shards, workers, False)

_x, _y, _n = next(iter(val_loader))
print('一個 batch:', _x.shape, _y[:8].tolist())


class OrdinalLabelSmoothingLoss(torch.nn.Module):
    """有序標籤平滑：將容錯機率只分配給左右相鄰 ±1 級（切合圍棋段位順序與競賽規則）。"""
    def __init__(self, num_classes=10, smoothing=0.05):
        super().__init__()
        target_table = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            target_table[i, i] = 1.0 - smoothing
            if i == 0:
                target_table[i, i + 1] = smoothing
            elif i == num_classes - 1:
                target_table[i, i - 1] = smoothing
            else:
                target_table[i, i - 1] = smoothing / 2.0
                target_table[i, i + 1] = smoothing / 2.0
        self.register_buffer('target_table', target_table)

    def forward(self, outputs, labels):
        return torch.nn.functional.cross_entropy(outputs, self.target_table[labels])


class ModelEMA:
    """
    模型權重指數移動平均 (Exponential Moving Average)：
    平滑 mini-batch 梯度震盪與 BatchNorm 統計量漂移，搭配動態暖身衰減率避免初期未收斂權重殘留。
    """
    def __init__(self, model, decay=0.9995):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

        # 預先按 (device, dtype) 分組綁定浮點數參數/BN統計量與整數計數器參照，實現單一融合核心高速更新
        ema_tensors = list(self.module.parameters()) + list(self.module.buffers())
        src_tensors = list(model.parameters()) + list(model.buffers())
        groups = defaultdict(lambda: ([], []))
        self._ema_nonfloat, self._src_nonfloat = [], []
        for e_t, s_t in zip(ema_tensors, src_tensors):
            if e_t.is_floating_point():
                groups[(e_t.device, e_t.dtype)][0].append(e_t)
                groups[(e_t.device, e_t.dtype)][1].append(s_t)
            else:
                self._ema_nonfloat.append(e_t)
                self._src_nonfloat.append(s_t)
        self._float_groups = list(groups.values())

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # 動態暖身衰減率：初期快速跟隨最新權重，隨步數推進平滑收斂至目標 decay (0.9995)
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        weight = 1.0 - d
        if hasattr(torch, '_foreach_lerp_'):
            for ema_list, src_list in self._float_groups:
                torch._foreach_lerp_(ema_list, src_list, weight)
        else:
            for ema_list, src_list in self._float_groups:
                for e_t, s_t in zip(ema_list, src_list):
                    e_t.lerp_(s_t, weight)
        for e_t, s_t in zip(self._ema_nonfloat, self._src_nonfloat):
            e_t.copy_(s_t)


model = GoRankResNet(num_classes=NUM_CLASSES).to(DEVICE)
ema_model = ModelEMA(model, decay=EMA_DECAY)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.3, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
)
criterion = OrdinalLabelSmoothingLoss(num_classes=NUM_CLASSES, smoothing=0.05).to(DEVICE)

print('參數量 {:,}'.format(sum(p.numel() for p in model.parameters())))
print('batch %d | lr %s (warmup %d ep) | ema %.4f | epochs %d'
      % (BATCH_SIZE, LEARNING_RATE, WARMUP_EPOCHS, EMA_DECAY, NUM_EPOCHS))


def validate(model, loader, criterion, use_tta=True):
    was_training = model.training
    model.eval()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if not use_tta:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    probs_sum = 0.0
                    loss_accum = 0.0
                    for flip in (False, True):
                        x_aug = torch.flip(inputs, dims=[-1]) if flip else inputs
                        for k in range(4):
                            x_rot = torch.rot90(x_aug, k, dims=[-2, -1]) if k > 0 else x_aug
                            out = model(x_rot)
                            loss_accum += criterion(out, labels).item()
                            probs_sum = probs_sum + out.softmax(dim=1)
                    loss = torch.tensor(loss_accum / 8.0)
                    _, predicted = torch.max(probs_sum, 1)
            loss_sum += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1

            # 累加混淆矩陣 (列=真實標籤, 欄=預測標籤)，不在每輪結束時印出
            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            np.add.at(cm, (y_true, y_pred), 1)
    if was_training:
        model.train()

    return (loss_sum / batches if batches else 0.0,
            100 * correct / total if total else 0.0,
            cm)


# 訓練期間不能中斷 kernel。
history = []
model.train()
best_val_acc = 0.0
best_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
train_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    total_loss, correct, total, steps = 0.0, 0, 0, 0

    for inputs, labels, _ in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # 圍棋棋盤 8 重對稱隨機增強（2種翻轉 × 4種旋轉 = 8種等價方向，於 GPU 整批高速運算）
        if torch.rand(1).item() < 0.5:
            inputs = torch.flip(inputs, dims=[-1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            inputs = torch.rot90(inputs, k, dims=[-2, -1])

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema_model.update(model)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        steps += 1
        if steps % 500 == 0:
            print('  ep%d step %d  loss %.4f  acc %.2f%%  %.1f 分'
                  % (epoch + 1, steps, total_loss / steps,
                     100 * correct / total, (time.time() - t0) / 60), flush=True)

    train_loss = total_loss / steps if steps else 0.0
    train_acc = 100 * correct / total if total else 0.0
    # 驗證與模型儲存統一採用平滑後的 EMA 模型權重
    val_loss, val_acc, val_cm = validate(ema_model.module, val_loader, criterion)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    epoch_minutes = (time.time() - t0) / 60

    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    })
    print('Epoch %d/%d  lr %.6f  |  train loss %.4f acc %.2f%%  |  val(EMA) loss %.4f acc %.2f%%  |  %.1f 分'
          % (epoch + 1, NUM_EPOCHS, current_lr, train_loss, train_acc, val_loss, val_acc,
             epoch_minutes), flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_cm = val_cm.copy()
        torch.save(ema_model.module.state_dict(), f'{CKPT_DIR}/best_model.pth')
        print('  新的最佳 val(EMA) acc %.2f%% —— 已存 best_model.pth' % best_val_acc)
    torch.save(ema_model.module.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))

total_train_minutes = (time.time() - train_start_time) / 60
pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
print('\n' + '=' * 60)
print('訓練完成，總耗時 %.1f 分（%.2f 小時），最佳 val acc: %.2f%%'
      % (total_train_minutes, total_train_minutes / 60, best_val_acc))

# --- 於訓練全部結束後，一次性輸出最佳模型的各等級分類指標與混淆矩陣 ---
tp = np.diag(best_cm)
pred_sum = best_cm.sum(axis=0)   # 每個類別被預測的總數 (Column sum)
true_sum = best_cm.sum(axis=1)   # 每個類別的真實樣本數 (Row sum)

precision_per_class = np.divide(tp, pred_sum, out=np.zeros_like(tp, dtype=np.float64), where=pred_sum > 0)
recall_per_class = np.divide(tp, true_sum, out=np.zeros_like(tp, dtype=np.float64), where=true_sum > 0)
f1_per_class = np.divide(
    2 * precision_per_class * recall_per_class,
    precision_per_class + recall_per_class,
    out=np.zeros_like(tp, dtype=np.float64),
    where=(precision_per_class + recall_per_class) > 0
)

metrics_df = pd.DataFrame({
    'Rank': LEVELS,
    'Precision (%)': np.round(precision_per_class * 100, 2),
    'Recall (%)': np.round(recall_per_class * 100, 2),
    'F1-score (%)': np.round(f1_per_class * 100, 2),
    'Support': true_sum
})
print('\n【各棋力等級分類指標】')
print(metrics_df.to_string(index=False))

cm_df = pd.DataFrame(
    best_cm,
    index=[f'True_{lv}' for lv in LEVELS],
    columns=[f'Pred_{lv}' for lv in LEVELS]
)
print('\n【混淆矩陣 (Confusion Matrix)】(列=真實等級, 欄=預測等級)')
print(cm_df.to_string())
print('=' * 60 + '\n')


import matplotlib.pyplot as plt

hist = pd.read_csv(f'{CKPT_DIR}/history.csv')
best = hist.loc[hist['val_acc'].idxmax()]

plt.figure(figsize=(9, 4))
plt.plot(hist['epoch'], hist['train_acc'], marker='o', ms=3, label='train acc')
plt.plot(hist['epoch'], hist['val_acc'], marker='o', ms=3, label='val acc')
plt.axvline(best['epoch'], ls='--', c='gray', lw=1)
plt.annotate('best %.2f%%' % best['val_acc'],
             xy=(best['epoch'], best['val_acc']),
             xytext=(best['epoch'] + 1.5, best['val_acc'] - 4),
             arrowprops=dict(arrowstyle='->', color='gray'))
plt.xlabel('epoch'); plt.ylabel('accuracy (%)')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('training_curve.png')
plt.close()
print("訓練曲線已儲存為 training_curve.png")

# --- 生成各棋力等級預測準確度折線圖 (rank_accuracy_curve.png) ---
rank_acc = recall_per_class * 100       # 各等級預測準確率 (Recall)
rank_prec = precision_per_class * 100   # 各等級精確率
rank_f1 = f1_per_class * 100            # 各等級 F1-score

plt.figure(figsize=(10, 5))
plt.plot(LEVELS, rank_acc, marker='o', lw=2, ms=6, label='Rank Accuracy / Recall (%)')
plt.plot(LEVELS, rank_prec, marker='s', lw=1.5, ms=5, ls='--', alpha=0.8, label='Precision (%)')
plt.plot(LEVELS, rank_f1, marker='^', lw=1.5, ms=5, ls='-.', alpha=0.8, label='F1-score (%)')
plt.axhline(best_val_acc, ls=':', c='red', lw=1.2, label=f'Overall Val Acc ({best_val_acc:.2f}%)')

for i, acc_val in enumerate(rank_acc):
    plt.annotate(f'{acc_val:.1f}%',
                 xy=(i, acc_val),
                 xytext=(0, 8),
                 textcoords='offset points',
                 ha='center', fontsize=9, fontweight='bold')

plt.title('Per-Rank Prediction Accuracy')
plt.xlabel('Rank')
plt.ylabel('Percentage (%)')
plt.ylim(0, max(rank_acc.max() + 12, 70))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rank_accuracy_curve.png', dpi=150)
plt.close()
print("各棋力等級預測準確度折線圖已儲存為 rank_accuracy_curve.png")

GPU_BATCH = 64
_infer_parser = SGFParseRankPrediction()


def sgf_to_features(args):
    """給 multiprocessing.Pool 用的頂層函式，接收 (sgf_content, color)。"""
    sgf_content, color = args
    feat, err = _infer_parser.features_from_sgf_content(sgf_content, color)
    return None if err is not None else feat


def predict(weights_path, out_path):
    net = GoRankResNet(num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval().to(DEVICE)

    rows = []
    with Pool(NPROC) as pool:
        # 兩份題目都要跑，結果串接成同一份 submission
        for split in ('public', 'private'):
            test = pd.read_csv(f'{TEST_DIR}/rank_prediction_test_{split}.csv')
            sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
            t0 = time.time()

            for idx, r in test.iterrows():
                # num_games 之後的 sgf_ 欄位是空字串，要濾掉，並配對對應的 color_i
                sgf_color_pairs = [
                    (r[c], r[c.replace('sgf_', 'color_')])
                    for c in sgf_cols
                    if pd.notna(r[c]) and str(r[c]).strip()
                ]
                feats = [f for f in pool.map(sgf_to_features, sgf_color_pairs) if f is not None]
                if not feats:
                    rows.append([r['question_id'], LEVELS[0]])
                    continue

                x = torch.from_numpy(np.stack(feats)).float()
                probs = []
                with torch.no_grad():
                    for i in range(0, len(x), GPU_BATCH):
                        bx = x[i:i + GPU_BATCH].to(DEVICE)
                        probs_sum = 0.0
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            for flip in (False, True):
                                x_aug = torch.flip(bx, dims=[-1]) if flip else bx
                                for k in range(4):
                                    x_rot = torch.rot90(x_aug, k, dims=[-2, -1]) if k > 0 else x_aug
                                    probs_sum = probs_sum + net(x_rot).softmax(1)
                        probs.append((probs_sum / 8.0).float().cpu().numpy())

                avg = np.concatenate(probs).mean(0)      # 同題多局的機率平均
                rows.append([r['question_id'], LEVELS[int(avg.argmax())]])
                if (idx + 1) % 100 == 0:
                    print('  %s %d/%d  %.0f 秒'
                          % (split, idx + 1, len(test), time.time() - t0), flush=True)
            print('  %s 完成 %d 題' % (split, len(test)), flush=True)

    sub = pd.DataFrame(rows, columns=['question_id', 'pred_rank'])
    sub.to_csv(out_path, index=False)
    print('寫出 %s（%d 列）' % (out_path, len(sub)))
    print('預測分布:', sub['pred_rank'].value_counts().to_dict())
    return sub


test_files = [
    f'{TEST_DIR}/rank_prediction_test_public.csv',
    f'{TEST_DIR}/rank_prediction_test_private.csv',
]
if all(os.path.isfile(path) for path in test_files):
    submission = predict(f'{CKPT_DIR}/best_model.pth', './submission_rank.csv')
    print("預測結果前 5 筆：")
    print(submission.head())
else:
    print('測試集檔案尚未提供，略過預測與 submission 輸出。')
