import random
import numpy as np
import torch
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, roc_curve
)

# 1) REPRODUCIBILITY
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# 2) PATHS & SAFE-LOAD
BASE_DIR = Path(r"C:\Users\PMYLS\How-to-download-ABIDE-Preprocessed-dataset-for-autism-detection")
DATA_DIR = BASE_DIR / "preprocessed_dataset"

X_PATH = DATA_DIR / "classification_data.pt"
assert X_PATH.exists(), f"{X_PATH} not found!"

def safe_load(p):
    return torch.load(p, weights_only=False)

raw = safe_load(X_PATH)
X_subj, y_subj = raw
if isinstance(y_subj, np.ndarray):
    y_subj = torch.from_numpy(y_subj)

print("Raw X_subj.shape:", X_subj.shape)
print("Raw y_subj.shape:", y_subj.shape)

# 2a) ALIGN y_subj → X_subj via feature_subject_ids.pt
ID_PATH = DATA_DIR / "feature_subject_ids.pt"
LBL_PATH = DATA_DIR / "binary_labels.pt"
assert ID_PATH.exists() and LBL_PATH.exists()
feat_ids = list(safe_load(ID_PATH))
labels_all = safe_load(LBL_PATH)
if isinstance(labels_all, np.ndarray):
    labels_all = torch.from_numpy(labels_all)

y_subj_aligned = torch.tensor([int(labels_all[sid]) for sid in feat_ids])
assert X_subj.shape[0] == y_subj_aligned.shape[0]

# 2b) FLATTEN windows → samples
n_subj, n_win, feat_dim = X_subj.shape
X_all = X_subj.reshape(n_subj * n_win, feat_dim)
y_all = y_subj_aligned.repeat_interleave(n_win)
print("Final X_all.shape:", X_all.shape, "y_all.shape:", y_all.shape)

# 3) CUSTOM DATASET
class WinDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.05):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_std = noise_std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[idx]

# 4) SIMPLE MLP
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5) 5-FOLD STRATIFIED CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all.numpy()), 1):
    print(f"\n--- Fold {fold}/5 ---")
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx].numpy(), y_all[te_idx].numpy()

    val_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold)
    tr2, val2 = next(val_split.split(X_tr, y_tr))
    X_train, X_val = X_tr[tr2], X_tr[val2]
    y_train, y_val = y_tr[tr2], y_tr[val2]

    cw = 1.0 / (np.bincount(y_train).astype(np.float32) ** 2)
    samp_w = cw[y_train]
    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    train_ds = WinDataset(X_train, torch.from_numpy(y_train), augment=True)
    val_ds   = WinDataset(X_val,   torch.from_numpy(y_val),   augment=False)
    test_ds  = WinDataset(X_te,    torch.from_numpy(y_te),    augment=False)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    model = MLP(feat_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_auc, best_state = 0.0, None
    for ep in range(1, 16):
        model.train()
        running = 0.0
        train_preds, train_labels = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            train_preds.append(torch.argmax(logits, dim=1).cpu())
            train_labels.append(yb.cpu())

        train_loss = running / len(train_loader.dataset)
        train_acc = accuracy_score(torch.cat(train_labels), torch.cat(train_preds))

        model.eval()
        val_scores, val_labels, val_preds = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)[:,1]
                preds = torch.argmax(logits, dim=1)
                val_scores.append(probs.cpu())
                val_labels.append(yb.cpu())
                val_preds.append(preds.cpu())

        yv = torch.cat(val_labels).numpy()
        sv = torch.cat(val_scores).numpy()
        pv = torch.cat(val_preds).numpy()
        val_acc = accuracy_score(yv, pv)
        auc_val = roc_auc_score(yv, sv)
        scheduler.step(auc_val)

        print(f"Ep{ep:02d} Loss:{train_loss:.4f} TrainAcc:{train_acc:.3f} ValAcc:{val_acc:.3f} ValAUC:{auc_val:.3f}")
        if auc_val > best_auc:
            best_auc = auc_val
            best_state = model.state_dict()

    print(f" Best Val AUC = {best_auc:.3f}")
    model.load_state_dict(best_state)

    fpr, tpr, thr = roc_curve(yv, sv)
    best_t = thr[np.argmax(tpr - fpr)]
    print(f" Threshold = {best_t:.3f}")

    model.eval()
    preds, labs, probs_list = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            ps = torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
            pr = (ps > best_t).astype(int)
            preds.append(pr)
            labs.append(yb.numpy())
            probs_list.append(ps)

    yp = np.concatenate(preds)
    yt = np.concatenate(labs)
    ps = np.concatenate(probs_list)

    test_acc = accuracy_score(yt, yp)
    auc  = roc_auc_score(yt, ps)
    cm   = confusion_matrix(yt, yp)
    prf  = precision_recall_fscore_support(yt, yp, zero_division=0)
    print(f" Test Acc:{test_acc:.3f} AUC:{auc:.3f}")
    print(" CM:\n", cm, "\nPRF:", prf)

    results.append((train_acc, val_acc, test_acc, auc, cm, prf))

# 6) SUMMARY
print("\n=== 5-Fold Summary ===")
for i, (train_acc, val_acc, test_acc, auc, cm, prf) in enumerate(results, 1):
    print(f"Fold{i} → TrainAcc:{train_acc:.3f} ValAcc:{val_acc:.3f} TestAcc:{test_acc:.3f} AUC:{auc:.3f}")
