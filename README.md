# 🛸 Drone Image AI Analysis Pipeline

> Python-based pipeline for detecting and classifying land-surface features from drone imagery using AI/ML techniques.

---

## 📁 Project Structure

```
drone_ai_assignment/
├── data/
│   ├── Drone_SAMPLE.tiff          ← Place your drone image here
│   └── generate_sample.py         ← Creates synthetic test image
├── src/
│   ├── data_loader.py             ← Load GeoTIFF / JPEG / PNG
│   ├── preprocessing.py           ← Resize · Denoise · Enhance · Normalise
│   ├── feature_extraction.py      ← 26-dim feature vector per patch
│   ├── model_training.py          ← Random Forest / SVM / CNN
│   ├── prediction.py              ← Tile-based inference + CSV export
│   └── visualization.py           ← 8 output figures
├── outputs/                       ← All generated outputs (auto-created)
├── main.py                        ← Pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone / download
git clone https://github.com/himanshu2285/Drone-Imagery-AI.git
cd Drone-Imagery-AI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option A — Use Drone_SAMPLE.tiff (your file)
```bash
# Place your file:
cp /path/to/Drone_SAMPLE.tiff data/

# Run the pipeline:
python main.py
```

### Option B — Generate a synthetic demo image
```bash
cd data && python generate_sample.py && cd ..
python main.py
```

### All CLI options
```bash
python main.py \
  --image        data/Drone_SAMPLE.tiff \
  --output_dir   outputs \
  --tile_size    64 \
  --resize       1024 \
  --model        rf \
  --n_estimators 200 \
  --alpha        0.5
```

| Argument | Default | Description |
|---|---|---|
| `--image` | `data/Drone_SAMPLE.tiff` | Input image path |
| `--output_dir` | `outputs` | Output folder |
| `--tile_size` | `64` | Patch size for classification |
| `--resize` | `1024` | Resize image side before processing |
| `--model` | `rf` | `rf` · `gb` · `svm` |
| `--n_estimators` | `200` | Trees for RF / GB |
| `--alpha` | `0.5` | Overlay transparency |
| `--use_cnn` | off | Enable CNN (PyTorch required) |
| `--no_augment` | off | Skip flip augmentation |

---

## 🧠 Technical Approach

### Step 1 — Data Loading (`data_loader.py`)
- Reads **GeoTIFF** (via `tifffile`), **JPEG**, **PNG**
- Handles multi-band imagery: band-first → band-last transpose
- Percentile stretching (2–98%) for uint16/float inputs → uint8
- `load_as_tiles()` for memory-efficient streaming of large images

### Step 2 — Preprocessing (`preprocessing.py`)
| Step | Method | Purpose |
|---|---|---|
| Resize | `cv2.resize` (aspect-preserving + reflect pad) | Fixed input size |
| Denoise | Bilateral filter (edge-preserving) | Remove sensor noise |
| Enhance | CLAHE on L-channel (LAB space) | Local contrast |
| Normalise | Min-max → float32 [0, 1] | Model-ready values |

### Step 3 — Feature Extraction (`feature_extraction.py`)
**26-dimensional feature vector per patch:**

| Block | Features | Count |
|---|---|---|
| Color statistics | R/G/B mean & std | 6 |
| HSV statistics | H-mean, S-mean, V-mean, S-std | 4 |
| GLCM Texture | contrast, dissimilarity, homogeneity, energy, correlation × 2 distances | 10 |
| Edge / Gradient | Sobel mean, Sobel std, Canny density | 3 |
| Vegetation indices | ExG, VARI | 2 |
| Shannon entropy | Grayscale image entropy | 1 |

### Step 4 — Model Training (`model_training.py`)
- **Random Forest** (primary): 200 trees, balanced class weights, parallel
- **Gradient Boosting** (alt): 200 trees, learning rate 0.1
- **SVM** (alt): RBF kernel, C=10, probability estimates
- **CNN** (optional): 3× ConvBnRelu → AdaptiveAvgPool → 2-layer FC head

**Land-Cover Classes:**
| ID | Class | Colour |
|---|---|---|
| 0 | Vegetation | Forest Green |
| 1 | Soil / Bare Ground | Earthy Brown |
| 2 | Water | Cobalt Blue |
| 3 | Built Structure | Brick Red |
| 4 | Road | Asphalt Grey |

### Step 5 — Label Generation (no ground-truth needed)
Colour heuristics per tile:
- **G ≫ R & B** → Vegetation
- **Dark + Blue ≥ Red** → Water
- **R ≫ G & B, moderate brightness** → Soil
- **Low saturation, moderate brightness** → Road
- **Default** → Built Structure

Replace `generate_labels_from_image()` with your annotated masks for production use.

### Step 6 — Prediction (`prediction.py`)
- Tiles the preprocessed image (configurable tile size / overlap)
- Extracts 26-dim features per tile
- Runs trained classifier → label + confidence
- Assembles `classification_map` (n_rows × n_cols)
- Upsamples to original image resolution (nearest-neighbour for labels)

### Step 7 — Visualisation (`visualization.py`)
| Output File | Description |
|---|---|
| `full_report.png` | 2×2 summary: original · overlay · colour map · confidence |
| `classification_overlay.png` | 3-panel comparison |
| `classification_map.png` | Pure colour-coded land-cover map |
| `confidence_map.png` | Green=high / Red=low confidence heatmap |
| `class_distribution.png` | Horizontal bar chart of % coverage |
| `tile_grid.png` | Tile-level grid with class colours |
| `feature_importances.png` | Top-20 Random Forest feature importances |
| `per_class_proba.png` | Per-class probability maps |
| `predictions.csv` | Per-tile: row, col, coords, class, confidence |
| `land_cover_model.pkl` | Saved trained model |

---

## 📦 Key Dependencies

| Library | Role |
|---|---|
| `opencv-python` | Image I/O, filtering, colour transforms |
| `scikit-image` | GLCM texture features, entropy |
| `scikit-learn` | Random Forest, SVM, metrics |
| `tifffile` | TIFF reading (multi-band, uint16) |
| `matplotlib` | All visualisations |
| `torch` *(optional)* | CNN classifier |

---

## 🔧 Extending

**Custom classes:** Edit `CLASS_NAMES` and `CLASS_RGB` in `model_training.py` and `visualization.py`.

**Real ground-truth:** Replace `generate_labels_from_image()` with a loader that reads your GeoJSON / shapefile / CSV annotations.

**NIR band:** If you have a 4-band RGBN image, pass the NIR band into `_vegetation_indices()` to enable NDVI.

**Large images:** Use `DroneImageLoader.load_as_tiles()` for memory-efficient streaming — no full image load required.