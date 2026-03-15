# Prediction Module

import os
import sys
import csv
import time
import numpy as np
import cv2

from feature_extraction import FeatureExtractor
from model_training     import LandCoverClassifier, CLASS_NAMES


class PredictionPipeline:

    def __init__(self,
                 classifier : LandCoverClassifier,
                 tile_size  : int = 64,
                 overlap    : int = 0):
        self.clf        = classifier
        self.tile_size  = tile_size
        self.overlap    = overlap
        self.extractor  = FeatureExtractor(patch_size=tile_size)

        # Results
        self.cls_map    : np.ndarray = None   # (n_rows, n_cols) int32
        self.conf_map   : np.ndarray = None   # (n_rows, n_cols) float32
        self.proba_map  : np.ndarray = None   # (n_rows, n_cols, n_classes)
        self.records    : list       = []     # per-tile CSV rows

    # Main method 

    def predict(self, image: np.ndarray) -> dict:
        
        img = self._ensure_uint8(image)
        H, W = img.shape[:2]
        step  = self.tile_size - self.overlap
        n_rows = int(np.ceil(H / step))
        n_cols = int(np.ceil(W / step))
        n_cls  = len(CLASS_NAMES)

        self._banner(H, W, n_rows, n_cols)

        cls_map   = np.zeros((n_rows, n_cols), dtype=np.int32)
        conf_map  = np.zeros((n_rows, n_cols), dtype=np.float32)
        proba_map = np.zeros((n_rows, n_cols, n_cls), dtype=np.float32)
        records   = []

        t0  = time.time()
        ri  = 0
        for y in range(0, H, step):
            ci = 0
            for x in range(0, W, step):
                # Extract tile
                y1   = min(y + self.tile_size, H)
                x1   = min(x + self.tile_size, W)
                tile = img[y:y1, x:x1]

                # Pad if needed
                ph = self.tile_size - tile.shape[0]
                pw = self.tile_size - tile.shape[1]
                if ph > 0 or pw > 0:
                    tile = np.pad(tile, ((0,ph),(0,pw),(0,0)), mode='reflect')

                # Resize to expected patch size
                tile = cv2.resize(tile, (self.tile_size, self.tile_size),
                                  interpolation=cv2.INTER_AREA)

                # Feature extraction + prediction
                feat  = self.extractor.extract(tile).reshape(1, -1)
                label = int(self.clf.predict(feat)[0])
                proba = self.clf.predict_proba(feat)[0]  # shape (n_classes,)
                conf  = float(proba.max())

                if ri < n_rows and ci < n_cols:
                    cls_map[ri, ci]     = label
                    conf_map[ri, ci]    = conf
                    proba_map[ri, ci]   = proba[:n_cls] if len(proba) >= n_cls \
                                          else np.pad(proba, (0, n_cls - len(proba)))

                records.append({
                    'tile_row'   : ri,
                    'tile_col'   : ci,
                    'pixel_y'    : y,
                    'pixel_x'    : x,
                    'label_id'   : label,
                    'label_name' : CLASS_NAMES.get(label, 'Unknown'),
                    'confidence' : round(conf, 4),
                })
                ci += 1
            ri += 1

        elapsed = time.time() - t0

        self.cls_map   = cls_map
        self.conf_map  = conf_map
        self.proba_map = proba_map
        self.records   = records

        self._summary(cls_map, n_rows * n_cols, elapsed)

        return {
            'classification_map': cls_map,
            'confidence_map'    : conf_map,
            'proba_map'         : proba_map,
            'records'           : records,
        }


    def get_label_image(self, original_shape: tuple) -> np.ndarray:
        self._require_prediction()
        H, W = original_shape[:2]
        return cv2.resize(
            self.cls_map.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

    def get_confidence_image(self, original_shape: tuple) -> np.ndarray:
        self._require_prediction()
        H, W = original_shape[:2]
        return cv2.resize(
            self.conf_map, (W, H),
            interpolation=cv2.INTER_LINEAR
        )

    def get_proba_image(self, original_shape: tuple) -> np.ndarray:
        self._require_prediction()
        H, W = original_shape[:2]
        n_cls = self.proba_map.shape[2]
        out   = np.zeros((H, W, n_cls), dtype=np.float32)
        for c in range(n_cls):
            out[:, :, c] = cv2.resize(
                self.proba_map[:, :, c], (W, H),
                interpolation=cv2.INTER_LINEAR
            )
        return out


    def export_csv(self, output_path: str = 'outputs/predictions.csv') -> str:
        
        self._require_prediction()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path)
                    else '.', exist_ok=True)

        fields = ['tile_row', 'tile_col', 'pixel_y', 'pixel_x',
                  'label_id', 'label_name', 'confidence']

        with open(output_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.records)

        print(f"[Predict] CSV exported → {output_path}  ({len(self.records)} rows)")
        return output_path

    @staticmethod
    def _ensure_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def _require_prediction(self):
        if self.cls_map is None:
            raise RuntimeError("[Predict] Call predict() before accessing results.")

    @staticmethod
    def _banner(H, W, n_rows, n_cols):
        print(f"\n[Predict] ══ Inference ══")
        print(f"  Image   : {H} × {W} px")
        print(f"  Grid    : {n_rows} rows × {n_cols} cols = {n_rows * n_cols} tiles")

    @staticmethod
    def _summary(cls_map, total, elapsed):
        uniq, cnts = np.unique(cls_map, return_counts=True)
        print(f"\n[Predict] ══ Results ({elapsed:.1f}s) ══")
        for u, c in zip(uniq, cnts):
            bar = '█' * int(c / total * 30)
            pct = c / total * 100
            print(f"  {CLASS_NAMES.get(u,'?'):18s}: {c:4d} tiles  {pct:5.1f}%  {bar}")
        print()



def run_prediction(image      : np.ndarray,
                   classifier : LandCoverClassifier,
                   tile_size  : int = 64,
                   output_dir : str = 'outputs') -> dict:
    
    pipe    = PredictionPipeline(classifier, tile_size=tile_size)
    results = pipe.predict(image)

    csv_path = os.path.join(output_dir, 'predictions.csv')
    pipe.export_csv(csv_path)

    results['label_image']      = pipe.get_label_image(image.shape)
    results['confidence_image'] = pipe.get_confidence_image(image.shape)
    results['proba_image']      = pipe.get_proba_image(image.shape)
    results['csv_path']         = csv_path
    results['pipeline']         = pipe

    return results


if __name__ == '__main__':
    print("Make prediction from main.py")