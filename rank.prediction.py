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
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  #shuffle the dataset
print('全量 %s 局' % format(len(df), ','))
print('等級分布', df['rank'].value_counts().to_dict())

df_val, df_train = df.iloc[:VAL_SIZE], df.iloc[VAL_SIZE:]
print('切分：訓練 %s / 驗證 %s' % (format(len(df_train), ','), format(len(df_val), ',')))


_parser = SGFParseRankPrediction()


def extract_worker(df_part, pattern, wid):
    parser = SGFParseRankPrediction()
    written = bad = 0
    t0 = time.time()
    with wds.ShardWriter(pattern, maxcount=PER_SHARD, verbose=0) as sink:
        for _, row in df_part.iterrows():
            feat, err = parser.features_from_sgf_content(row['sgf_content'])
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


t0 = time.time()
extract_split(df_val, 'val')
extract_split(df_train, 'train')
print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))


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


model = GoRankResNet(num_classes=NUM_CLASSES).to(DEVICE)
# 【更動 3】加入 weight_decay 防止過擬合
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 【更動 4】加入動態學習率調整 (Cosine Annealing)
# 它會在 T_max (也就是 NUM_EPOCHS) 的過程中，把學習率平滑地降到 eta_min (1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

criterion = torch.nn.CrossEntropyLoss() # 計算交叉熵損失

print('參數量 {:,}'.format(sum(p.numel() for p in model.parameters())))
print('batch %d | lr %s | epochs %d' % (BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS))


def validate(model, loader, criterion): # 驗證函式
    model.eval()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
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
    model.train()
    return (loss_sum / batches if batches else 0.0,
            100 * correct / total if total else 0.0)


history = []
model.train()
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    total_loss, correct, total, steps = 0.0, 0, 0, 0

    for inputs, labels, _ in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # 反向傳播
        optimizer.step() # 更新參數

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        steps += 1
        if steps % 500 == 0:
            print('  ep%d step %d  loss %.4f  acc %.2f%%  %.1f 分'
                  % (epoch + 1, steps, total_loss / steps,
                     100 * correct / total, (time.time() - t0) / 60), flush=True)

    train_acc = 100 * correct / total if total else 0.0
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # 【更動 5】每個 Epoch 結束後，呼叫排程器讓學習率「走一步」平滑下降
    scheduler.step()
    
    # 讀取當前的學習率數值，準備印出來觀察
    current_lr = scheduler.get_last_lr()[0]
    
    history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc, 'lr': current_lr})
    
    # 這裡的 print 多加了 LR %.6f，讓你能親眼看到學習率變小的過程
    print('Epoch %d/%d  train acc %.2f%%  |  val loss %.4f acc %.2f%%  |  LR %.6f  |  %.1f 分'
          % (epoch + 1, NUM_EPOCHS, train_acc, val_loss, val_acc, current_lr,
             (time.time() - t0) / 60), flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pth')
        print('  新的最佳 val acc %.2f%% —— 已存 best_model.pth' % best_val_acc)
    torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))

pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
print('訓練完成，最佳 val acc %.2f%%' % best_val_acc)


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


def sgf_to_features(sgf_content):
    """給 multiprocessing.Pool 用的頂層函式。"""
    feat, err = _infer_parser.features_from_sgf_content(sgf_content)
    return None if err is not None else feat


def predict(weights_path, out_path):
    net = GoRankResNet(num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval().to(DEVICE)

    rows = []
    with Pool(NPROC) as pool:
        for split in ('public', 'private'):
            test = pd.read_csv(f'{TEST_DIR}/rank_prediction_test_{split}.csv')
            sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
            t0 = time.time()

            for idx, r in test.iterrows():
                sgfs = [r[c] for c in sgf_cols
                        if pd.notna(r[c]) and str(r[c]).strip()]
                feats = [f for f in pool.map(sgf_to_features, sgfs) if f is not None]
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

#submission = predict(f'{CKPT_DIR}/best_model.pth', './submission_rank.csv')
#print("預測結果前 5 筆：")
#print(submission.head())