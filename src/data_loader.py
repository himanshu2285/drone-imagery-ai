# File to load the data

import os
import sys
import numpy as np
import cv2
from PIL import Image

# This is Optional rasterio for full GeoTIFF metadata
try:
    import rasterio
    _RASTERIO = True
except ImportError:
    _RASTERIO = False

try:
    import tifffile
    _TIFFFILE = True
except ImportError:
    _TIFFFILE = False


SUPPORTED_EXT = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}


class DroneImageLoader:

    def __init__(self, image_path: str, verbose: bool = True):
        self.image_path = os.path.abspath(image_path)
        self.verbose    = verbose

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(
                f"[DataLoader] File not found: {self.image_path}\n"
                f"  → Place Drone_SAMPLE.tiff inside the data/ folder."
            )

        self._ext = os.path.splitext(self.image_path)[1].lower()
        if self._ext not in SUPPORTED_EXT:
            raise ValueError(
                f"[DataLoader] Unsupported extension '{self._ext}'. "
                f"Supported: {SUPPORTED_EXT}"
            )

        self.image    : np.ndarray = None   # (H, W, 3) uint8
        self.metadata : dict       = {}


    def load(self) -> np.ndarray:
        # To Load the image and return a (H, W, 3) uint8 RGB numpy array.
        self._log(f"Loading: {os.path.basename(self.image_path)}")

        if self._ext in ('.tif', '.tiff'):
            self.image = self._load_tiff()
        else:
            self.image = self._load_standard()

        self._build_metadata()
        self._print_summary()
        return self.image

    def load_as_tiles(self, tile_size: int = 256, overlap: int = 32) -> list:
        """Split the loaded image into overlapping tiles."""
        if self.image is None:
            self.load()

        H, W  = self.image.shape[:2]
        step  = tile_size - overlap
        tiles = []
        row   = 0

        for y in range(0, H, step):
            col = 0
            for x in range(0, W, step):
                y1   = min(y + tile_size, H)
                x1   = min(x + tile_size, W)
                tile = self.image[y:y1, x:x1].copy()

                # Pad with reflection to reach full tile_size
                ph = tile_size - tile.shape[0]
                pw = tile_size - tile.shape[1]
                if ph > 0 or pw > 0:
                    tile = np.pad(tile, ((0, ph), (0, pw), (0, 0)),
                                  mode='reflect')

                tiles.append({
                    'tile': tile, 'row': row, 'col': col,
                    'y0': y, 'x0': x, 'y1': y1, 'x1': x1,
                })
                col += 1
            row += 1

        self._log(f"Tiling → {len(tiles)} tiles "
                  f"({row} rows × {col} cols, "
                  f"tile_size={tile_size}, overlap={overlap})")
        return tiles

    def tile_grid_shape(self, tile_size: int, overlap: int) -> tuple:
        """Return (n_rows, n_cols) of the tile grid."""
        if self.image is None:
            self.load()
        H, W = self.image.shape[:2]
        step = tile_size - overlap
        return (int(np.ceil(H / step)), int(np.ceil(W / step)))

    # For Private loaders

    def _load_tiff(self) -> np.ndarray:
        """Load TIFF: tifffile → PIL → OpenCV fallback chain."""
        if _TIFFFILE:
            try:
                arr = tifffile.imread(self.image_path)
                self._log(f"  [tifffile] raw shape={arr.shape}, dtype={arr.dtype}")
                return self._to_rgb_uint8(arr)
            except Exception as e:
                self._log(f"  [tifffile] failed ({e}) → trying PIL …")

        try:
            pil = Image.open(self.image_path).convert('RGB')
            return np.array(pil, dtype=np.uint8)
        except Exception as e:
            self._log(f"  [PIL] failed ({e}) → trying OpenCV …")

        arr = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"[DataLoader] Cannot read: {self.image_path}")
        return self._to_rgb_uint8(arr)

    def _load_standard(self) -> np.ndarray:
        """Load JPEG / PNG via OpenCV (BGR → RGB)."""
        arr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if arr is None:
            raise IOError(f"[DataLoader] OpenCV failed: {self.image_path}")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


    @staticmethod
    def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
        """
        Robustly convert any raw ndarray to (H, W, 3) uint8 RGB.
        Handles:  (C,H,W), (H,W), (H,W,1), (H,W,4), uint16, float32
        """
        # Band-first → band-last
        if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] in (1,3,4,5):
            arr = np.transpose(arr, (1, 2, 0))

        # Grayscale → RGB
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr]*3, axis=-1)

        # Drop extra bands (keep first 3)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        # Stretch to uint8 [0,255] using 2-98 percentile
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)

    def _build_metadata(self):
        H, W, C = self.image.shape
        self.metadata = {
            'file'        : os.path.basename(self.image_path),
            'path'        : self.image_path,
            'size_kb'     : os.path.getsize(self.image_path) // 1024,
            'height_px'   : H,
            'width_px'    : W,
            'channels'    : C,
            'dtype'       : str(self.image.dtype),
            'pixel_min'   : int(self.image.min()),
            'pixel_max'   : int(self.image.max()),
            'pixel_mean'  : round(float(self.image.mean()), 2),
            'memory_mb'   : round(self.image.nbytes / 1_048_576, 2),
        }

    def _print_summary(self):
        if not self.verbose:
            return
        m  = self.metadata
        ln = "─" * 50
        print(f"\n  {ln}")
        print(f"  ✔  Image loaded successfully")
        print(f"  {ln}")
        print(f"  File        : {m['file']}")
        print(f"  Disk size   : {m['size_kb']} KB")
        print(f"  Dimensions  : {m['height_px']} H × {m['width_px']} W × {m['channels']} C")
        print(f"  Dtype       : {m['dtype']}")
        print(f"  Pixel range : [{m['pixel_min']}, {m['pixel_max']}]  "
              f"mean = {m['pixel_mean']}")
        print(f"  Memory      : {m['memory_mb']} MB")
        print(f"  {ln}\n")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[DataLoader] {msg}")


def load_drone_image(image_path: str, verbose: bool = True) -> tuple:
    """
    One-liner to load a drone image.

    Returns
    -------
    (image: np.ndarray (H,W,3) uint8,  metadata: dict)
    """
    loader = DroneImageLoader(image_path, verbose=verbose)
    image  = loader.load()
    return image, loader.metadata


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/Drone_SAMPLE.tiff'
    img, meta = load_drone_image(path)
    print("Metadata:", meta)
    loader = DroneImageLoader(path, verbose=False)
    loader.load()
    tiles = loader.load_as_tiles(256, 32)
    print(f"Tile count : {len(tiles)}")
    print(f"Tile shape : {tiles[0]['tile'].shape}")