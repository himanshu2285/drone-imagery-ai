# Module to Train the Model

import os
import sys
import numpy as np
import joblib
import cv2

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay)



CLASS_NAMES = {
    0: 'Vegetation',
    1: 'Soil',
    2: 'Water',
    3: 'Built Structure',
    4: 'Road',
}
NUM_CLASSES = len(CLASS_NAMES)


# Label generation 
def heuristic_label(patch: np.ndarray) -> int:
    
    if patch.dtype != np.uint8:
        patch = np.clip(patch * 255, 0, 255).astype(np.uint8)

    R = float(patch[:, :, 0].mean())
    G = float(patch[:, :, 1].mean())
    B = float(patch[:, :, 2].mean())
    brightness = (R + G + B) / 3.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = float(hsv[:, :, 1].mean())

    # 1. Vegetation
    if G > R * 1.12 and G > B * 1.12 and G > 55:
        return 0

    # 2. Water  — dark-ish, blue ≥ red
    if brightness < 100 and B >= R * 0.95:
        return 2

    # 3. Soil  — reddish / warm, not too dark
    if R > G * 1.08 and R > B * 1.08 and 60 < brightness < 175:
        return 1

    # 4. Road  — grey (low sat), moderate-to-high brightness
    if sat < 45 and brightness > 70:
        return 4

    # 5. Built structure
    return 3


def generate_labels_from_image(image: np.ndarray,
                                tile_size: int = 64) -> tuple:
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    H, W = image.shape[:2]
    patches, labels = [], []

    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):
            p = image[y:y + tile_size, x:x + tile_size]
            if p.shape[:2] == (tile_size, tile_size):
                patches.append(p)
                labels.append(heuristic_label(p))

    return patches, np.array(labels, dtype=np.int32)


def augment_patches(patches: list, labels: np.ndarray) -> tuple:
    aug_p, aug_l = [], []
    for p, l in zip(patches, labels):
        aug_p += [p, np.fliplr(p), np.flipud(p)]
        aug_l += [l, l, l]
    return aug_p, np.array(aug_l, dtype=np.int32)



class LandCoverClassifier:
    # Sklearn-based land-cover classifier.

    def __init__(self, model_type: str = 'rf', n_estimators: int = 200):
        """
        model: 'rf' | 'gb' | 'svm'
        """
        self.model_type   = model_type
        self.n_estimators = n_estimators
        self.scaler       = StandardScaler()
        self.model        = self._build(model_type, n_estimators)
        self.is_trained   = False
        self.classes_     = list(CLASS_NAMES.keys())
        self.metrics_     = {}


    def train(self, X: np.ndarray, y: np.ndarray,
              val_size: float = 0.2) -> dict:
        print(f"\n[Model] ══ Training {self.model_type.upper()} ══")
        print(f"  Samples  : {len(X)}")
        print(f"  Features : {X.shape[1]}")
        self._print_class_dist(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=42
        )

        X_tr_s  = self.scaler.fit_transform(X_tr)
        X_val_s = self.scaler.transform(X_val)

        self.model.fit(X_tr_s, y_tr)
        self.is_trained = True

        # Metrics
        y_pred_tr  = self.model.predict(X_tr_s)
        y_pred_val = self.model.predict(X_val_s)

        tr_acc  = accuracy_score(y_tr,  y_pred_tr)
        val_acc = accuracy_score(y_val, y_pred_val)

        present_classes = sorted(np.unique(np.concatenate([y_tr, y_val])).tolist())
        target_names    = [CLASS_NAMES[c] for c in present_classes]

        report   = classification_report(y_val, y_pred_val,
                                          labels=present_classes,
                                          target_names=target_names,
                                          zero_division=0)
        conf_mat = confusion_matrix(y_val, y_pred_val, labels=present_classes)

        print(f"\n  Train accuracy      : {tr_acc:.4f}")
        print(f"  Validation accuracy : {val_acc:.4f}")
        print(f"\n  Classification Report (Validation):\n{report}")
        print(f"  Confusion Matrix:\n{conf_mat}\n")

        self.metrics_ = {
            'train_accuracy'  : tr_acc,
            'val_accuracy'    : val_acc,
            'report'          : report,
            'confusion_matrix': conf_mat,
            'present_classes' : present_classes,
        }

        if hasattr(self.model, 'feature_importances_'):
            self.metrics_['feature_importances'] = self.model.feature_importances_

        return self.metrics_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.  Returns (N,) int array."""
        self._require_trained()
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.  Returns (N, C) float array."""
        self._require_trained()
        return self.model.predict_proba(self.scaler.transform(X))

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       n_splits: int = 5) -> dict:
        """
        Stratified K-fold cross-validation (uses scaled features).
        """
        X_s    = self.scaler.fit_transform(X)
        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_s, y, cv=skf,
                                  scoring='accuracy', n_jobs=-1)
        result = {'cv_mean': scores.mean(), 'cv_std': scores.std(),
                  'cv_scores': scores.tolist()}
        print(f"  CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        return result

    def save(self, path: str = 'outputs/land_cover_model.pkl'):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'classes': self.classes_, 'model_type': self.model_type}, path)
        print(f"[Model] Saved → {path}")

    def load(self, path: str = 'outputs/land_cover_model.pkl'):
        obj = joblib.load(path)
        self.model      = obj['model']
        self.scaler     = obj['scaler']
        self.classes_   = obj.get('classes', list(CLASS_NAMES.keys()))
        self.model_type = obj.get('model_type', 'rf')
        self.is_trained = True
        print(f"[Model] Loaded ← {path}")

    # ── Private 

    @staticmethod
    def _build(model_type: str, n_estimators: int):
        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
            )
        elif model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf', C=10, gamma='scale',
                probability=True, class_weight='balanced',
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type '{model_type}'. "
                             "Choose: 'rf', 'gb', 'svm'.")

    @staticmethod
    def _print_class_dist(y: np.ndarray):
        uniq, cnts = np.unique(y, return_counts=True)
        print("  Class distribution:")
        for u, c in zip(uniq, cnts):
            bar = '█' * int(c / max(cnts) * 20)
            print(f"    {CLASS_NAMES.get(u, u):18s} [{c:5d}]  {bar}")

    def _require_trained(self):
        if not self.is_trained:
            raise RuntimeError("[Model] Not trained yet — call train() first.")



# CNN classifier (PyTorch)
class CNNClassifier:

    def __init__(self, num_classes: int = NUM_CLASSES, patch_size: int = 64):
        self.num_classes = num_classes
        self.patch_size  = patch_size
        self.model       = None
        self.device      = None
        self._init()

    def _init(self):
        try:
            import torch
            import torch.nn as nn
            self.torch  = torch
            self.nn     = nn
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            class _Net(nn.Module):
                def __init__(self, nc):
                    super().__init__()
                    self.enc = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4)),
                    )
                    self.cls = nn.Sequential(
                        nn.Dropout(0.4),
                        nn.Linear(128 * 16, 256), nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, nc),
                    )
                def forward(self, x):
                    return self.cls(self.enc(x).flatten(1))

            self.model = _Net(self.num_classes).to(self.device)
            print(f"[CNN] Initialised on {self.device}")
        except ImportError:
            print("[CNN] PyTorch not installed — CNN classifier unavailable.")

    def train(self, patches: list, labels: np.ndarray,
              epochs: int = 25, batch_size: int = 32, lr: float = 1e-3):
        if self.model is None:
            raise RuntimeError("PyTorch not available.")
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X = np.stack([cv2.resize(p, (self.patch_size, self.patch_size))
                      .astype(np.float32) / 255.0 for p in patches])
        X = torch.tensor(X.transpose(0, 3, 1, 2))
        y = torch.tensor(labels, dtype=torch.long)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        tr_dl  = DataLoader(TensorDataset(X_tr, y_tr),  batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        opt  = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = self.nn.CrossEntropyLoss()
        hist = []

        for ep in range(1, epochs + 1):
            self.model.train()
            tot = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward(); opt.step()
                tot += loss.item()
            sch.step()

            self.model.eval()
            cor = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    cor += (self.model(xb).argmax(1) == yb).sum().item()
            acc = cor / len(y_val)
            hist.append({'epoch': ep, 'loss': tot / len(tr_dl), 'val_acc': acc})
            if ep % 5 == 0 or ep == 1:
                print(f"  Epoch {ep:3d}/{epochs}  "
                      f"loss={tot/len(tr_dl):.4f}  val_acc={acc:.4f}")
        return hist

    def predict(self, patches: list) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PyTorch not available.")
        import torch
        X = np.stack([cv2.resize(p, (self.patch_size, self.patch_size))
                      .astype(np.float32) / 255.0 for p in patches])
        X = torch.tensor(X.transpose(0, 3, 1, 2)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1).cpu().numpy()

    def save(self, path: str = 'outputs/cnn_model.pt'):
        if self.model is None: return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.torch.save(self.model.state_dict(), path)
        print(f"[CNN] Saved → {path}")



if __name__ == '__main__':
    np.random.seed(42)
    X_demo = np.random.rand(300, 26).astype(np.float32)
    y_demo = np.random.randint(0, NUM_CLASSES, 300)

    clf = LandCoverClassifier('rf', n_estimators=100)
    m   = clf.train(X_demo, y_demo)
    p   = clf.predict(X_demo[:5])
    print("Predictions:", p, [CLASS_NAMES[i] for i in p])