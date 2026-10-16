import sys
import glob
import os
import time
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import webdataset as wds
from torch.utils.data import DataLoader

from utils import SGFParseRankPrediction
from network import GoRankResNet


# --- 在程式最頂端啟動總計時器
program_start_time = time.time()
# --- 加入日誌記錄功能 ---
class Logger(object):
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8') 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("output_log.txt")
# ------------------------

print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| webdataset', wds.__version__)


# 官方資料放置位置：training/ 與 tests/ 依官方發佈的結構擺放
TRAIN_DIR = './training'
TEST_DIR = './tests'
FEAT_DIR = './rank-features'
CKPT_DIR = './rank-trained-models'

LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK_TO_ID = {lv: i for i, lv in enumerate(LEVELS)}
ID_TO_RANK = {i: lv for lv, i in RANK_TO_ID.items()}

# CPU 版本的 multiprocessing.Pool
NPROC = 8              
PER_SHARD = 1000       
SEED = 42
VAL_SIZE = 100_000     

# 【更動 1】保持較小的 Batch Size 以增加防呆雜訊
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# 【更動 2】保持充足的 Epochs 讓模型收斂
NUM_EPOCHS = 20
NUM_CLASSES = len(LEVELS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(CKPT_DIR, exist_ok=True)
print('device', DEVICE)
RANK_TO_ID


df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{TRAIN_DIR}/train_*.csv'))],
               ignore_index=True)
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print('全量 %s 局' % format(len(df), ','))
print('等級分布', df['rank'].value_counts().to_dict())

# 記錄所有執白 (color='W') 的棋局 key，供舊版特徵分片在讀取時動態對調黑白視角
_is_white = df['color'].astype(str).str.strip().str.upper() == 'W'
WHITE_GAME_KEYS = set(
    (df.loc[_is_white, 'rank'].astype(str) + '_' + df.loc[_is_white, 'game_id'].astype(str)).tolist()
)
SWAP_INDICES = torch.tensor(SGFParseRankPrediction.build_perspective_swap_indices(), dtype=torch.long)
print('執白 (W) 棋局數: %s / 全量 %s' % (format(len(WHITE_GAME_KEYS), ','), format(len(df), ',')))

df_val, df_train = df.iloc[:VAL_SIZE], df.iloc[VAL_SIZE:]
print('切分：訓練 %s / 驗證 %s' % (format(len(df_train), ','), format(len(df_val), ',')))


_parser = SGFParseRankPrediction()


def extract_worker(df_part, pattern, wid):
    parser = SGFParseRankPrediction()
    written = bad = 0
    t0 = time.time()
    with wds.ShardWriter(pattern, maxcount=PER_SHARD, verbose=0) as sink:
        for _, row in df_part.iterrows():
            color_str = str(row['color']).strip().upper()
            feat, err = parser.features_from_sgf_content(row['sgf_content'], color=color_str)
            if err is not None:
                bad += 1
                continue
            sink.write({
                '__key__': '%s_%s' % (row['rank'], row['game_id']),
                'features.npy': feat,
                'label.txt': str(row['rank']),
                'rank_idx.txt': str(RANK_TO_ID[row['rank']]),
                'game_name.txt': str(row['game_id']),
                'color.txt': color_str,
            })
            written += 1
            if written % 20000 == 0:
                print('  [w%d] %d/%d  %.0f 分'
                      % (wid, written, len(df_part), (time.time() - t0) / 60),
                      flush=True)
    print('  [w%d] 寫入 %d 筆（解析失敗 %d）%.1f 分'
          % (wid, written, bad, (time.time() - t0) / 60), flush=True)


def extract_split(df_part, tag):
    outdir = f'{FEAT_DIR}/{tag}'
    os.makedirs(outdir, exist_ok=True)

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
    if avg < 10 * 1024:
        raise SystemExit('%s: 分片平均僅 %.0f bytes，判定為空分片' % (tag, avg))


IN_CHANNELS = 353


def _existing_shards_match_channels(feat_dir, expected_channels=IN_CHANNELS):
    train_files = sorted(glob.glob(f'{feat_dir}/train/*.tar.gz'))
    if not train_files:
        return False
    try:
        sample = next(iter(wds.WebDataset([train_files[0]]).decode()))
        ch = sample['features.npy'].shape[0]
        if ch != expected_channels:
            print(f'偵測到舊版特徵分片（通道數 {ch} != {expected_channels}），準備清理並重新抽取 353 通道特徵...')
            for old_f in glob.glob(f'{feat_dir}/*/*.tar.gz'):
                os.remove(old_f)
            return False
        return True
    except Exception as e:
        print(f'檢查現有特徵分片時發生異常 ({e})，將重新抽取特徵...')
        return False


if not _existing_shards_match_channels(FEAT_DIR, IN_CHANNELS):
    t0 = time.time()
    extract_split(df_val, 'val')
    extract_split(df_train, 'train')
    print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))
else:
    print(f'偵測到 {FEAT_DIR} 已存在 {IN_CHANNELS} 通道特徵分片，直接跳過特徵抽取步驟！')


# 【更動 8】新增隨機變換函式：利用數學張量操作產生 8 種對稱視角
def augment_features(features):
    """隨機對 19x19 的盤面進行 8 種圍棋對稱變換之一"""
    k = torch.randint(0, 8, (1,)).item()
    
    # 前 4 種 (0~3)：單純旋轉 0, 90, 180, 270 度
    # 後 4 種 (4~7)：先水平翻轉，再旋轉 0, 90, 180, 270 度
    if k >= 4:
        features = torch.flip(features, dims=[-1]) # 水平翻轉
    
    rot = k % 4
    if rot > 0:
        features = torch.rot90(features, k=rot, dims=[-2, -1]) # 對最後兩個空間維度旋轉
        
    return features


def make_loader(shards, workers, shuffle):
    ds = wds.WebDataset(shards, shardshuffle=1 if shuffle else 0)
    if shuffle:
        ds = ds.shuffle(1000)
        
    def process_sample(sample):
        features = torch.from_numpy(sample['features.npy']).float()

        # 若讀取的是未含 color.txt 的舊版特徵分片，且該局目標玩家執白 (W)，動態將黑白通道對調為「我方/對手」視角
        if 'color.txt' not in sample and sample['__key__'] in WHITE_GAME_KEYS:
            features = features[SWAP_INDICES]
        
        # 【更動 9】攔截機制：如果這批資料是準備拿來訓練的 (shuffle=True)，就套用資料增強
        if shuffle:
            features = augment_features(features)
            
        label = torch.tensor(int(sample['rank_idx.txt']), dtype=torch.long)
        game_name = sample['game_name.txt']
        return features, label, game_name

    ds = ds.decode().map(process_sample).batched(BATCH_SIZE, partial=True)
    return DataLoader(ds, batch_size=None, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


def pick_workers(val_shards, candidate=4):
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
model = GoRankResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
# 【更動 3】加入 weight_decay 防止過擬合
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 【更動 4】加入動態學習率調整 (Cosine Annealing)
# 它會在 T_max (也就是 NUM_EPOCHS) 的過程中，把學習率平滑地降到 eta_min (1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

criterion = torch.nn.CrossEntropyLoss()

print('參數量 {:,}'.format(sum(p.numel() for p in model.parameters())))
print('batch %d | lr %s | epochs %d' % (BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS))


def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss_sum += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1

            # 累加混淆矩陣 (列=真實標籤, 欄=預測標籤)
            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            np.add.at(cm, (y_true, y_pred), 1)

    model.train()
    return (loss_sum / batches if batches else 0.0,
            100 * correct / total if total else 0.0,
            cm)


history = []
model.train()
best_val_acc = 0.0
best_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    total_loss, correct, total, steps = 0.0, 0, 0, 0

    for inputs, labels, _ in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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
    val_loss, val_acc, val_cm = validate(model, val_loader, criterion)
    
    # 【更動 5】每個 Epoch 結束後，呼叫排程器讓學習率「走一步」平滑下降
    scheduler.step()
    
    # 讀取當前的學習率數值，準備印出來觀察
    current_lr = scheduler.get_last_lr()[0]
    
    # 寫入每回合結束前的 train_loss、val_loss、train_acc、val_acc
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    })
    
    print('Epoch %d/%d  train loss %.4f acc %.2f%%  |  val loss %.4f acc %.2f%%  |  LR %.6f  |  %.1f 分'
          % (epoch + 1, NUM_EPOCHS, train_loss, train_acc, val_loss, val_acc, current_lr,
             (time.time() - t0) / 60), flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_cm = val_cm.copy()
        torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pth')
        print('  新的最佳 val acc %.2f%% —— 已存 best_model.pth' % best_val_acc)
    torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))

pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
print('\n' + '=' * 60)
print('訓練完成，最佳 val acc: %.2f%%' % best_val_acc)

# --- 計算並輸出最佳模型的混淆矩陣、Precision、Recall、F1-score ---
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

macro_precision = precision_per_class.mean() * 100
macro_recall = recall_per_class.mean() * 100
macro_f1 = f1_per_class.mean() * 100

total_samples = true_sum.sum()
weighted_precision = (np.sum(precision_per_class * true_sum) / total_samples * 100) if total_samples > 0 else 0.0
weighted_recall = (np.sum(recall_per_class * true_sum) / total_samples * 100) if total_samples > 0 else 0.0
weighted_f1 = (np.sum(f1_per_class * true_sum) / total_samples * 100) if total_samples > 0 else 0.0

print(f'Macro Precision   : {macro_precision:.2f}%')
print(f'Macro Recall      : {macro_recall:.2f}%')
print(f'Macro F1-score    : {macro_f1:.2f}%')
print(f'Weighted Precision: {weighted_precision:.2f}%')
print(f'Weighted Recall   : {weighted_recall:.2f}%')
print(f'Weighted F1-score : {weighted_f1:.2f}%')

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
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); 
plt.savefig('training_curve.png') 
print("訓練曲線已儲存為 training_curve.png")

# 結算並印出整個程式的總耗時
total_elapsed = time.time() - program_start_time
total_hours = int(total_elapsed // 3600)
total_minutes = int((total_elapsed % 3600) // 60)
print(f'整個程式總執行時長：{total_hours} 小時 {total_minutes} 分鐘')




#end

GPU_BATCH = 64
_infer_parser = SGFParseRankPrediction()


def sgf_to_features(args):
    """給 multiprocessing.Pool 用的頂層函式，接收 (sgf_content, color) 或單一 sgf_content。"""
    if isinstance(args, tuple):
        sgf_content, color = args
    else:
        sgf_content, color = args, 'B'
    feat, err = _infer_parser.features_from_sgf_content(sgf_content, color=color)
    return None if err is not None else feat


def predict(weights_path, out_path):
    net = GoRankResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval().to(DEVICE)

    rows = []
    with Pool(NPROC) as pool:
        for split in ('public', 'private'):
            test = pd.read_csv(f'{TEST_DIR}/rank_prediction_test_{split}.csv')
            sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
            t0 = time.time()

            for idx, r in test.iterrows():
                sgf_color_pairs = []
                for c in sgf_cols:
                    if pd.notna(r[c]) and str(r[c]).strip():
                        suffix = c[len('sgf_'):]
                        color_col = f'color_{suffix}'
                        color = str(r[color_col]).strip() if (color_col in r and pd.notna(r[color_col])) else 'B'
                        sgf_color_pairs.append((r[c], color))

                feats = [f for f in pool.map(sgf_to_features, sgf_color_pairs) if f is not None]
                if not feats:
                    rows.append([r['question_id'], LEVELS[0]])
                    continue

                x = torch.from_numpy(np.stack(feats)).float()
                probs = []
                with torch.no_grad():
                    for i in range(0, len(x), GPU_BATCH):
                        out = net(x[i:i + GPU_BATCH].to(DEVICE))
                        probs.append(out.softmax(1).cpu().numpy())

                avg = np.concatenate(probs).mean(0)      
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

# 等拿到測試集後，把下面三行註解拿掉即可產生提交檔案：
#submission = predict(f'{CKPT_DIR}/best_model.pth', './submission_rank.csv')
#print("預測結果前 5 筆：")
#print(submission.head())
