# Visualization Module

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from model_training import CLASS_NAMES

CLASS_RGB = {
    0: (34,  140,  34),   # Vegetation     → Forest Green
    1: (180, 130,  70),   # Soil           → Earthy Brown
    2: (30,  120, 220),   # Water          → Cobalt Blue
    3: (200,  80,  80),   # Built Struct   → Brick Red
    4: (110, 110, 110),   # Road           → Asphalt Grey
}

_FALLBACK = [(255,165,0),(148,0,211),(0,200,200),(255,100,180)]


def _rgb(label: int) -> tuple:
    return CLASS_RGB.get(label, _FALLBACK[label % len(_FALLBACK)])


def _rgb_norm(label: int) -> tuple:
    """Return normalised [0,1] RGB tuple for matplotlib."""
    return tuple(c / 255.0 for c in _rgb(label))


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _colour_map_image(label_img: np.ndarray) -> np.ndarray:
    """Convert (H, W) int32 label array to (H, W, 3) uint8 RGB."""
    out = np.zeros((*label_img.shape[:2], 3), dtype=np.uint8)
    for lbl, rgb in CLASS_RGB.items():
        out[label_img == lbl] = rgb
    return out


def _legend_patches(present: list) -> list:
    return [mpatches.Patch(color=_rgb_norm(l),
                            label=CLASS_NAMES.get(l, f"Class {l}"))
            for l in present]


def _savefig(fig, path: str, dpi: int = 150):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Viz] Saved → {path}")


#1. For Full Report
def plot_full_report(original_image   : np.ndarray,
                     label_image      : np.ndarray,
                     confidence_image : np.ndarray,
                     save_path        : str = 'outputs/full_report.png'):
    
    orig      = _to_uint8(original_image)
    cmap_img  = _colour_map_image(label_image)
    overlay   = cv2.addWeighted(orig, 0.45, cmap_img, 0.55, 0)
    present   = sorted(np.unique(label_image).tolist())

    fig = plt.figure(figsize=(18, 14), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.05)

    titles = ['Original Drone Image', 'Classification Overlay',
              'Classification Map',   'Confidence Heat-Map']
    panels = [orig, overlay, cmap_img, None]
    axes   = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    style = dict(fontsize=13, fontweight='bold', color='white', pad=8)

    for ax, title, panel in zip(axes[:3], titles[:3], panels[:3]):
        ax.imshow(panel)
        ax.set_title(title, **style)
        ax.axis('off')

    ax_conf = axes[3]
    im = ax_conf.imshow(confidence_image, cmap='RdYlGn', vmin=0, vmax=1)
    ax_conf.set_title(titles[3], **style)
    ax_conf.axis('off')
    cbar = fig.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.03)
    cbar.set_label('Confidence', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Shared legend
    fig.legend(handles=_legend_patches(present),
               loc='lower center', ncol=len(present),
               fontsize=11, frameon=True,
               facecolor='#2d2d44', edgecolor='white',
               labelcolor='white')

    fig.suptitle('Drone Image AI Analysis — Full Report',
                 fontsize=17, fontweight='bold', color='white', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    _savefig(fig, save_path, dpi=150)


# 2. Classification Overlay
def plot_classification_overlay(original_image : np.ndarray,
                                 label_image    : np.ndarray,
                                 alpha          : float = 0.5,
                                 save_path      : str   = 'outputs/classification_overlay.png'):
    
    orig     = _to_uint8(original_image)
    cmap_img = _colour_map_image(label_image)
    blended  = cv2.addWeighted(orig, 1 - alpha, cmap_img, alpha, 0)
    present  = sorted(np.unique(label_image).tolist())

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')

    panels = [orig, cmap_img, blended]
    titles = ['Original Drone Image', 'Classification Map',
              f'Overlay  (α = {alpha:.1f})']

    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight='bold',
                     color='white', pad=8)
        ax.axis('off')

    fig.legend(handles=_legend_patches(present),
               loc='lower center', ncol=len(present),
               fontsize=11, frameon=True,
               facecolor='#1c2333', edgecolor='#4a90d9',
               labelcolor='white')

    plt.suptitle('Land-Cover Classification', fontsize=16,
                 fontweight='bold', color='white')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    _savefig(fig, save_path)


# 3. Standalone Classification Map
def plot_classification_map(label_image : np.ndarray,
                             save_path  : str = 'outputs/classification_map.png'):
    """Full-size colour-coded land-cover map with legend."""
    cmap_img = _colour_map_image(label_image)
    present  = sorted(np.unique(label_image).tolist())

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.imshow(cmap_img)
    ax.axis('off')
    ax.set_title('Land-Cover Classification Map', fontsize=15,
                  fontweight='bold', color='white', pad=10)

    fig.legend(handles=_legend_patches(present),
               loc='lower center', ncol=len(present),
               fontsize=12, frameon=True,
               facecolor='#1c2333', edgecolor='#4a90d9',
               labelcolor='white')

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _savefig(fig, save_path)


# 4. Confidence Heatmap
def plot_confidence_map(confidence_image : np.ndarray,
                         save_path        : str = 'outputs/confidence_map.png'):
    fig, ax = plt.subplots(figsize=(11, 9), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    im = ax.imshow(confidence_image, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    ax.axis('off')
    ax.set_title('Prediction Confidence Map', fontsize=15,
                  fontweight='bold', color='white', pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Confidence Score', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    mean_conf = float(confidence_image.mean())
    ax.text(0.02, 0.02, f'Mean confidence: {mean_conf:.3f}',
            transform=ax.transAxes, color='white', fontsize=10,
            bbox=dict(facecolor='#1c2333', alpha=0.8, pad=4))

    plt.tight_layout()
    _savefig(fig, save_path)


# 5. Class Distribution Bar Chart

def plot_class_distribution(label_image : np.ndarray,
                              save_path  : str = 'outputs/class_distribution.png'):
    """Horizontal bar chart showing % coverage per class."""
    labels, counts = np.unique(label_image, return_counts=True)
    names   = [CLASS_NAMES.get(l, f"Class {l}") for l in labels]
    colors  = [_rgb_norm(l) for l in labels]
    pcts    = counts / counts.sum() * 100

    # Sort descending
    order  = np.argsort(pcts)[::-1]
    names  = [names[i]  for i in order]
    pcts   = pcts[order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    bars = ax.barh(names, pcts, color=colors, edgecolor='#333344',
                   linewidth=0.8, height=0.6)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', color='white', fontsize=11,
                fontweight='bold')

    ax.set_xlabel('Coverage (%)', color='white', fontsize=12)
    ax.set_title('Land-Cover Class Distribution', fontsize=15,
                  fontweight='bold', color='white', pad=12)
    ax.set_xlim(0, max(pcts) * 1.18)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444455')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#444455')
    ax.xaxis.label.set_color('white')
    plt.tick_params(axis='both', colors='white')

    plt.tight_layout()
    _savefig(fig, save_path)


# 6. Tile Grid
def plot_tile_grid(classification_map : np.ndarray,
                   save_path          : str = 'outputs/tile_grid.png'):
    """Visualise the tile-level classification grid (zoomed + grid lines)."""
    cmap_img = _colour_map_image(classification_map)
    n_r, n_c = classification_map.shape

    # Zoom factor so the image is at least 400 px wide
    zoom = max(1, 400 // max(n_r, n_c, 1))
    h_z  = n_r * zoom
    w_z  = n_c * zoom
    zoomed = cv2.resize(cmap_img, (w_z, h_z), interpolation=cv2.INTER_NEAREST)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.imshow(zoomed)

    # Draw grid
    for r in range(n_r + 1):
        ax.axhline(r * zoom - 0.5, color='white', linewidth=0.4, alpha=0.6)
    for c in range(n_c + 1):
        ax.axvline(c * zoom - 0.5, color='white', linewidth=0.4, alpha=0.6)

    ax.set_title('Tile-Level Classification Grid', fontsize=14,
                  fontweight='bold', color='white', pad=8)
    ax.axis('off')

    present = sorted(np.unique(classification_map).tolist())
    fig.legend(handles=_legend_patches(present),
               loc='lower center', ncol=len(present),
               fontsize=11, frameon=True,
               facecolor='#1c2333', edgecolor='#4a90d9',
               labelcolor='white')

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _savefig(fig, save_path)


# 7. Feature Importances
def plot_feature_importances(importances    : np.ndarray,
                               feature_names : list  = None,
                               top_n         : int   = 20,
                               save_path     : str   = 'outputs/feature_importances.png'):
    """Horizontal bar chart of top-N feature importances (Random Forest)."""
    importances = np.array(importances)
    idx   = np.argsort(importances)[::-1][:top_n]
    vals  = importances[idx]

    if feature_names and len(feature_names) == len(importances):
        flabels = [feature_names[i] for i in idx]
    else:
        flabels = [f"F{i}" for i in idx]

    # Colour by importance magnitude
    norm   = plt.Normalize(vals.min(), vals.max())
    colors = plt.cm.YlOrRd(norm(vals))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    bars = ax.barh(flabels[::-1], vals[::-1], color=colors[::-1],
                   edgecolor='#333344', linewidth=0.6, height=0.7)

    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', color='white', fontsize=9)

    ax.set_xlabel('Importance Score', color='white', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)',
                  fontsize=14, fontweight='bold', color='white', pad=10)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444455')
    ax.xaxis.label.set_color('white')
    plt.tick_params(axis='both', colors='white')
    plt.tight_layout()
    _savefig(fig, save_path)


# 8. Per-Class Probability Panels
def plot_per_class_probabilities(proba_image : np.ndarray,
                                  present_classes: list = None,
                                  save_path  : str = 'outputs/per_class_proba.png'):
    n_cls = proba_image.shape[2]
    if present_classes is None:
        present_classes = list(range(n_cls))

    n_show = len(present_classes)
    ncols  = min(n_show, 3)
    nrows  = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5 * nrows),
                              facecolor='#0d1117')
    axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]

    for i, cls_id in enumerate(present_classes):
        ax   = axes_flat[i]
        prob = proba_image[:, :, cls_id] if cls_id < proba_image.shape[2] \
               else np.zeros(proba_image.shape[:2])

        
        rgb = np.array(_rgb(cls_id)) / 255.0
        cmap = LinearSegmentedColormap.from_list(
            f'cls{cls_id}',
            [(0.05, 0.05, 0.1), rgb],
            N=256
        )
        im = ax.imshow(prob, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
                      color='white', fontsize=12, fontweight='bold')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Hide unused axes
    for ax in axes_flat[n_show:]:
        ax.set_visible(False)

    fig.suptitle('Per-Class Probability Maps', fontsize=15,
                  fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    _savefig(fig, save_path)


def generate_all_outputs(original_image   : np.ndarray,
                          label_image      : np.ndarray,
                          confidence_image : np.ndarray,
                          proba_image      : np.ndarray      = None,
                          cls_map          : np.ndarray      = None,
                          feature_importances : np.ndarray   = None,
                          feature_names    : list            = None, # type: ignore
                          output_dir       : str             = 'outputs',
                          alpha            : float           = 0.5) -> list:
    os.makedirs(output_dir, exist_ok=True)
    p   = lambda name: os.path.join(output_dir, name)
    saved = []

    def _run(fn, *args, **kw):
        try:
            fn(*args, **kw)
            saved.append(kw.get('save_path', ''))
        except Exception as e:
            print(f"[Viz] Warning — {fn.__name__} failed: {e}")

    _run(plot_full_report,
         original_image, label_image, confidence_image,
         save_path=p('full_report.png'))

    _run(plot_classification_overlay,
         original_image, label_image, alpha=alpha,
         save_path=p('classification_overlay.png'))

    _run(plot_classification_map,
         label_image,
         save_path=p('classification_map.png'))

    _run(plot_confidence_map,
         confidence_image,
         save_path=p('confidence_map.png'))

    _run(plot_class_distribution,
         label_image,
         save_path=p('class_distribution.png'))

    if cls_map is not None:
        _run(plot_tile_grid,
             cls_map,
             save_path=p('tile_grid.png'))

    if feature_importances is not None:
        _run(plot_feature_importances,
             feature_importances,
             feature_names=feature_names,
             save_path=p('feature_importances.png'))

    if proba_image is not None:
        present = sorted(np.unique(label_image).tolist())
        _run(plot_per_class_probabilities,
             proba_image,
             present_classes=present,
             save_path=p('per_class_proba.png'))

    return saved


if __name__ == '__main__':
    # Smoke-test
    H = W = 256
    fake_orig  = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    fake_label = np.random.randint(0, 5,   (H, W),    dtype=np.int32)
    fake_conf  = np.random.rand(H, W).astype(np.float32)
    fake_proba = np.random.dirichlet(np.ones(5), H * W).reshape(H, W, 5).astype(np.float32)

    saved = generate_all_outputs(
        fake_orig, fake_label, fake_conf,
        proba_image=fake_proba,
        cls_map=np.random.randint(0, 5, (8, 8), dtype=np.int32),
        feature_importances=np.random.rand(26),
        output_dir='../outputs'
    )
    print("Saved files:", saved)