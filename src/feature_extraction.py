# Module for extracting features

import sys
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy



# Constants
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS     = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
GLCM_LEVELS    = 32          # quantise to 32 grey levels for speed

FEATURE_NAMES = (
    ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std']           # 6
    + ['H_mean', 'S_mean', 'V_mean', 'S_std']                            # 4
    + [f"GLCM_{p}_d{d}" for d in GLCM_DISTANCES for p in GLCM_PROPS]   # 10
    + ['Sobel_mean', 'Sobel_std', 'Canny_density']                       # 3
    + ['ExG', 'VARI']                                                    # 2
    + ['Entropy']                                                         # 1
)                                                                        # = 26



class FeatureExtractor:
    # Extracts a fixed-length feature vector from one image patch.

    def __init__(self, patch_size: int = 64):
        self.patch_size = patch_size

    def extract(self, patch: np.ndarray) -> np.ndarray:
        # Extract all features from a single patch.

        u8  = self._to_uint8(patch)
        f32 = self._to_float32(patch)

        feats = np.concatenate([
            self._color_stats(f32),          # 6
            self._hsv_stats(u8),             # 4
            self._glcm_texture(u8),          # 10
            self._edge_features(u8),         # 3
            self._vegetation_indices(f32),   # 2
            self._entropy(u8),               # 1
        ])
        return feats.astype(np.float32)

    def extract_batch(self, patches: list, verbose: bool = False) -> np.ndarray:
        # Extract features for a list / array of patches.
        all_feats = []
        n = len(patches)
        for i, p in enumerate(patches):
            all_feats.append(self.extract(p))
            if verbose and (i + 1) % 100 == 0:
                print(f"  [FeatureExtractor] {i+1}/{n} patches processed …")
        mat = np.stack(all_feats, axis=0)
        if verbose:
            print(f"  [FeatureExtractor] Done. Matrix shape: {mat.shape}")
        return mat

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)

    @property
    def feature_names(self) -> list:
        return FEATURE_NAMES.copy()

    # Feature blocks
    def _color_stats(self, patch_f32: np.ndarray) -> np.ndarray:
        means = patch_f32.mean(axis=(0, 1))   # (3,)
        stds  = patch_f32.std(axis=(0, 1))    # (3,)
        return np.concatenate([means, stds])

    def _hsv_stats(self, patch_u8: np.ndarray) -> np.ndarray:
        # HSV statistics — captures hue (colour identity), 4 features: [H_mean, S_mean, V_mean, S_std]
        hsv = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        h_mean = hsv[:, :, 0].mean() / 180.0   # normalise Hue to [0,1]
        s_mean = hsv[:, :, 1].mean() / 255.0
        v_mean = hsv[:, :, 2].mean() / 255.0
        s_std  = hsv[:, :, 1].std()  / 255.0
        return np.array([h_mean, s_mean, v_mean, s_std], dtype=np.float32)

    def _glcm_texture(self, patch_u8: np.ndarray) -> np.ndarray:
        
        gray  = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2GRAY)
        gray_q = (gray // (256 // GLCM_LEVELS)).astype(np.uint8)
        gray_q = np.clip(gray_q, 0, GLCM_LEVELS - 1)

        feats = []
        for dist in GLCM_DISTANCES:
            glcm = graycomatrix(
                gray_q,
                distances=[dist],
                angles=GLCM_ANGLES,
                levels=GLCM_LEVELS,
                symmetric=True,
                normed=True,
            )
            for prop in GLCM_PROPS:
                val = float(graycoprops(glcm, prop).mean())
                feats.append(val)

        return np.array(feats, dtype=np.float32)

    def _edge_features(self, patch_u8: np.ndarray) -> np.ndarray:
        
        gray = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2GRAY)

        # Sobel gradient magnitude
        sx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)

        sobel_mean = float(mag.mean()) / 255.0
        sobel_std  = float(mag.std())  / 255.0

        # Canny edge density
        edges         = cv2.Canny(gray, 40, 120)
        canny_density = float(edges.sum()) / (edges.size * 255.0)

        return np.array([sobel_mean, sobel_std, canny_density], dtype=np.float32)

    def _vegetation_indices(self, patch_f32: np.ndarray) -> np.ndarray:
        R = patch_f32[:, :, 0]
        G = patch_f32[:, :, 1]
        B = patch_f32[:, :, 2]

        exg  = float((2.0 * G - R - B).mean())

        denom = G + R - B
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        vari  = float(np.clip(((G - R) / denom).mean(), -1.0, 1.0))

        return np.array([exg, vari], dtype=np.float32)

    def _entropy(self, patch_u8: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2GRAY)
        ent  = float(shannon_entropy(gray))
        return np.array([ent], dtype=np.float32)

    # Static helpers

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_float32(img: np.ndarray) -> np.ndarray:
        if img.dtype in (np.float32, np.float64):
            return img.astype(np.float32)
        return img.astype(np.float32) / 255.0



def extract_features_from_tiles(tiles: list,
                                 patch_size: int = 64,
                                 verbose: bool = True) -> tuple:
    
    extractor = FeatureExtractor(patch_size=patch_size)
    feats, info = [], []

    n = len(tiles)
    for i, t in enumerate(tiles):
        tile = cv2.resize(t['tile'], (patch_size, patch_size),
                          interpolation=cv2.INTER_AREA)
        feats.append(extractor.extract(tile))
        info.append((t['row'], t['col'], t['y0'], t['x0']))
        if verbose and (i + 1) % 50 == 0:
            print(f"  [Features] {i+1}/{n} tiles …")

    if verbose:
        print(f"  [Features] Extraction complete. Matrix: {len(feats)} × {extractor.n_features}")
    return np.stack(feats), info


if __name__ == '__main__':
    ext = FeatureExtractor(patch_size=64)
    print(f"Feature count   : {ext.n_features}")
    print(f"Feature names   : {ext.feature_names}")

    dummy   = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    feat    = ext.extract(dummy)
    print(f"\nSample feature vector ({len(feat)} dims):")
    for name, val in zip(ext.feature_names, feat):
        print(f"  {name:25s}: {val:.4f}")