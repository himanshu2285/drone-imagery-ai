# Preprocessing Module

import sys
import numpy as np
import cv2
from PIL import Image


class ImagePreprocessor:

    def __init__(self, image: np.ndarray):
        """
        Parameters
        ----------
        image : np.ndarray  (H, W, 3) uint8 RGB
        """
        if image is None or image.ndim < 2:
            raise ValueError("[Preprocessor] Received an invalid image array.")
        self._original = image.copy()
        self.image     = image.copy()  # working copy
        print(f"[Preprocessor] Initialised  shape={image.shape}  dtype={image.dtype}")
        
    # Step 1: Resizing
    def resize(self,
               target: tuple = (1024, 1024),
               keep_aspect: bool = True,
               interpolation=cv2.INTER_AREA) -> 'ImagePreprocessor':
        
        h, w = self.image.shape[:2]
        tw, th = target

        if keep_aspect:
            scale  = min(tw / w, th / h)
            new_w  = int(w * scale)
            new_h  = int(h * scale)
            shrunk = cv2.resize(self.image, (new_w, new_h), interpolation=interpolation)

            pad_t = (th - new_h) // 2
            pad_b = th - new_h - pad_t
            pad_l = (tw - new_w) // 2
            pad_r = tw - new_w - pad_l

            self.image = cv2.copyMakeBorder(
                shrunk, pad_t, pad_b, pad_l, pad_r,
                cv2.BORDER_REFLECT_101
            )
        else:
            self.image = cv2.resize(self.image, (tw, th), interpolation=interpolation)

        print(f"[Preprocessor] Resized   {h}×{w} → {self.image.shape[0]}×{self.image.shape[1]}")
        return self

    def make_tiles(self,
                   tile_size: int = 256,
                   overlap:   int = 32) -> list:
       
        img  = self._as_uint8(self.image)
        H, W = img.shape[:2]
        step  = tile_size - overlap
        tiles = []
        row   = 0

        for y in range(0, H, step):
            col = 0
            for x in range(0, W, step):
                y1   = min(y + tile_size, H)
                x1   = min(x + tile_size, W)
                tile = img[y:y1, x:x1].copy()

                ph = tile_size - tile.shape[0]
                pw = tile_size - tile.shape[1]
                if ph > 0 or pw > 0:
                    tile = np.pad(tile, ((0, ph), (0, pw), (0, 0)), mode='reflect')

                tiles.append({
                    'tile': tile,
                    'row' : row, 'col': col,
                    'y0'  : y,   'x0' : x,
                    'y1'  : y1,  'x1' : x1,
                })
                col += 1
            row += 1

        print(f"[Preprocessor] Tiling    → {len(tiles)} tiles  "
              f"(size={tile_size}, overlap={overlap})")
        return tiles

    # Step 2: Noise Removal
    def denoise(self,
                method: str = 'bilateral',
                **kw) -> 'ImagePreprocessor':
        img = self._as_uint8(self.image)

        if method == 'gaussian':
            k = kw.get('ksize', 3)
            s = kw.get('sigma', 0)
            out = cv2.GaussianBlur(img, (k, k), s)

        elif method == 'median':
            k   = kw.get('ksize', 3)
            out = cv2.medianBlur(img, k)

        elif method == 'bilateral':
            d  = kw.get('d', 9)
            sc = kw.get('sigma_color', 75)
            ss = kw.get('sigma_space', 75)
            out = cv2.bilateralFilter(img, d, sc, ss)

        elif method == 'nlmeans':
            h = kw.get('h', 10)
            out = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

        else:
            raise ValueError(f"[Preprocessor] Unknown denoise method: '{method}'")

        self.image = out.astype(self.image.dtype) if self.image.dtype != np.uint8 else out
        print(f"[Preprocessor] Denoised  method={method}")
        return self

    # Step 3 – Contrast Enhancement
    def enhance(self,
                clip_limit: float = 2.0,
                tile_grid: tuple  = (8, 8)) -> 'ImagePreprocessor':
        img = self._as_uint8(self.image)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        L_eq  = clahe.apply(L)

        lab_eq = cv2.merge([L_eq, A, B])
        out    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        self.image = out
        print(f"[Preprocessor] Enhanced  CLAHE clip={clip_limit} grid={tile_grid}")
        return self

    def sharpen(self, strength: float = 0.5) -> 'ImagePreprocessor':
        img = self._as_uint8(self.image)
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        out = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
        self.image = out
        print(f"[Preprocessor] Sharpened strength={strength}")
        return self

    # Step 4 – Normalisation
    def normalise(self, method: str = 'minmax') -> 'ImagePreprocessor':
        img = self.image.astype(np.float32)

        if method == 'minmax':
            lo = img.min(); hi = img.max()
            self.image = ((img - lo) / (hi - lo + 1e-8)).astype(np.float32)

        elif method == 'zscore':
            out = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[2]):
                mu  = img[:, :, c].mean()
                std = img[:, :, c].std() + 1e-8
                out[:, :, c] = (img[:, :, c] - mu) / std
            self.image = out

        elif method == 'uint8':
            lo = np.percentile(img, 1); hi = np.percentile(img, 99)
            img = np.clip((img - lo) / (hi - lo + 1e-8) * 255, 0, 255)
            self.image = img.astype(np.uint8)

        else:
            raise ValueError(f"[Preprocessor] Unknown normalise method: '{method}'")

        print(f"[Preprocessor] Normalised method={method}  "
              f"range=[{self.image.min():.3f}, {self.image.max():.3f}]")
        return self


    # Full pipeline
    def run_pipeline(self,
                     target_size:   tuple = (1024, 1024),
                     denoise_method: str  = 'bilateral',
                     norm_method:    str  = 'minmax',
                     do_enhance:     bool = True,
                     do_sharpen:     bool = False) -> np.ndarray:
        print("\n[Preprocessor] ══ Pipeline Start ══")
        self.resize(target_size) \
            .denoise(denoise_method)
        if do_enhance:
            self.enhance()
        if do_sharpen:
            self.sharpen()
        self.normalise(norm_method)
        print(f"[Preprocessor] ══ Pipeline Done   output={self.image.shape} ══\n")
        return self.image


    # Helpers
    def get(self) -> np.ndarray:
        """Return current image state."""
        return self.image

    def get_uint8(self) -> np.ndarray:
        """Return current image as uint8 [0,255]."""
        return self._as_uint8(self.image)

    def reset(self) -> 'ImagePreprocessor':
        """Reset to original (un-preprocessed) image."""
        self.image = self._original.copy()
        print("[Preprocessor] Reset to original.")
        return self

    @staticmethod
    def _as_uint8(img: np.ndarray) -> np.ndarray:
        """Convert float [0,1] → uint8 [0,255] if needed, else return as-is."""
        if img.dtype in (np.float32, np.float64):
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        return img.astype(np.uint8)

    @staticmethod
    def patch_for_model(patch: np.ndarray, size: int = 64) -> np.ndarray:
        p = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        if p.dtype == np.uint8:
            return p.astype(np.float32) / 255.0
        return p.astype(np.float32)



if __name__ == '__main__':
    sys.path.insert(0, '.')
    from data_loader import load_drone_image

    path = sys.argv[1] if len(sys.argv) > 1 else '../data/Drone_SAMPLE.tiff'
    raw, _ = load_drone_image(path)

    proc   = ImagePreprocessor(raw)
    result = proc.run_pipeline(target_size=(1024, 1024))
    print("Output shape :", result.shape)
    print("Output dtype :", result.dtype)
    print("Value range  :", result.min(), "–", result.max())