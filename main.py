"""
Main File for Drone Image AI Analysis Pipeline
"""

import argparse
import os
import sys
import time
import numpy as np
 
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data_loader import DroneImageLoader, load_drone_image
from preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from model_training import (LandCoverClassifier, CNNClassifier, generate_labels_from_image, augment_patches, CLASS_NAMES)
from prediction import PredictionPipeline, run_prediction
from visualization import generate_all_outputs


# cli
def parse_args():
    p = argparse.ArgumentParser(
        description='Drone Image AI Analysis Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--image',        type=str,   default=None,
                   help='Path to input TIFF/JPG/PNG. Defaults to data/Drone_SAMPLE.tiff')
    p.add_argument('--output_dir',   type=str,   default='outputs')
    p.add_argument('--tile_size',    type=int,   default=64)
    p.add_argument('--resize',       type=int,   default=1024,
                   help='Resize image to this side length before processing')
    p.add_argument('--model',        type=str,   default='rf',
                   choices=['rf', 'gb', 'svm'])
    p.add_argument('--n_estimators', type=int,   default=200)
    p.add_argument('--alpha',        type=float, default=0.5,
                   help='Overlay transparency [0–1]')
    p.add_argument('--use_cnn',      action='store_true',
                   help='Also train a CNN (requires PyTorch)')
    p.add_argument('--no_augment',   action='store_true',
                   help='Skip data augmentation')
    return p.parse_args()


# Helpers
def _header(text: str):
    w = 60
    print(f"\n{'═'*w}")
    print(f"  {text}")
    print(f"{'═'*w}")


def _step(n: int, total: int, text: str):
    print(f"\nStep {n}/{total}: {text}")


def _done(elapsed: float):
    print(f"Done  ({elapsed:.1f}s)")


def _find_image(arg_path) -> str:
    """Resolve image path"""
    candidates = [
        arg_path,
        os.path.join(ROOT, 'data', 'Drone_SAMPLE.tiff'),
        os.path.join(ROOT, 'data', 'drone_image.tiff'),
        os.path.join(ROOT, 'data', 'drone_image.tif'),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No drone image found. "
        "Place Drone_SAMPLE.tiff in the data/ folder or pass --image <path>."
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    TOTAL_STEPS = 8
    t_global    = time.time()

    _header("Drone Image AI Analysis Pipeline  —  Starting")

    # Step 1: Load 
    _step(1, TOTAL_STEPS, "Data Loading")
    t0         = time.time()
    image_path = _find_image(args.image)
    print(f"  Image path : {image_path}")

    loader   = DroneImageLoader(image_path, verbose=True)
    raw_img  = loader.load()                   # (H, W, 3) uint8 RGB
    meta     = loader.metadata
    _done(time.time() - t0)

    # Step 2: Preprocess
    _step(2, TOTAL_STEPS, "Preprocessing  (resize → denoise → enhance → normalise)")
    t0 = time.time()

    preprocessor = ImagePreprocessor(raw_img)
    proc_f32     = preprocessor.run_pipeline(
        target_size    = (args.resize, args.resize),
        denoise_method = 'bilateral',
        norm_method    = 'minmax',
        do_enhance     = True,
        do_sharpen     = False,
    )
    # uint8 version for tiling & visualisation
    proc_u8 = preprocessor.get_uint8()
    _done(time.time() - t0)

    # Step 3: Generate training labels
    _step(3, TOTAL_STEPS, "Generating heuristic training labels")
    t0 = time.time()

    patches, labels = generate_labels_from_image(proc_u8, tile_size=args.tile_size)
    print(f"  Patches generated : {len(patches)}")
    _done(time.time() - t0)

    # Step 4: Augmentation 
    _step(4, TOTAL_STEPS, "Data Augmentation")
    t0 = time.time()

    if not args.no_augment:
        patches, labels = augment_patches(patches, labels)
        print(f"  Patches after augmentation : {len(patches)}")
    else:
        print("  Augmentation skipped.")
    _done(time.time() - t0)

    # Step 5: Feature Extraction
    _step(5, TOTAL_STEPS, "Feature Extraction  (26 features per patch)")
    t0  = time.time()
    ext = FeatureExtractor(patch_size=args.tile_size)
    X   = ext.extract_batch(patches, verbose=True)
    y   = labels
    print(f"  Feature matrix : {X.shape}  Labels: {y.shape}")
    _done(time.time() - t0)

    # Step 6: Train
    _step(6, TOTAL_STEPS, f"Training {args.model.upper()} Classifier")
    t0  = time.time()
    clf = LandCoverClassifier(args.model, n_estimators=args.n_estimators)
    metrics = clf.train(X, y, val_size=0.2)
    clf.save(os.path.join(args.output_dir, 'land_cover_model.pkl'))

    # Here i used CNN for optional training
    if args.use_cnn:
        print("\n  [Optional] Training CNN …")
        cnn = CNNClassifier(num_classes=len(CLASS_NAMES),
                             patch_size=args.tile_size)
        if cnn.model is not None:
            cnn.train(patches, labels, epochs=30)
            cnn.save(os.path.join(args.output_dir, 'cnn_model.pt'))
    _done(time.time() - t0)

    # Step 7: Predict
    _step(7, TOTAL_STEPS, "Running Tile-by-Tile Prediction")
    t0      = time.time()
    pipe    = PredictionPipeline(clf, tile_size=args.tile_size, overlap=0)
    results = pipe.predict(proc_u8)

    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    pipe.export_csv(csv_path)

    label_img = pipe.get_label_image(proc_u8.shape)
    conf_img  = pipe.get_confidence_image(proc_u8.shape)
    proba_img = pipe.get_proba_image(proc_u8.shape)
    _done(time.time() - t0)

    # Step 8: Visualise
    _step(8, TOTAL_STEPS, "Generating Visualisations")
    t0 = time.time()

    saved = generate_all_outputs(
        original_image      = proc_u8,
        label_image         = label_img,
        confidence_image    = conf_img,
        proba_image         = proba_img,
        cls_map             = results['classification_map'],
        feature_importances = metrics.get('feature_importances'),
        feature_names       = ext.feature_names,
        output_dir          = args.output_dir,
        alpha               = args.alpha,
    )
    _done(time.time() - t0)

    # F Summary
    elapsed = time.time() - t_global
    _header("Pipeline Complete!")
    print(f"  Total time        : {elapsed:.1f}s")
    print(f"  Image loaded      : {meta['file']}  "
          f"({meta['width_px']}×{meta['height_px']})")
    print(f"  Training patches  : {len(patches)}")
    print(f"  Feature dims      : {X.shape[1]}")
    print(f"  Validation acc    : {metrics['val_accuracy']:.4f}")
    print(f"  Tiles predicted   : {len(results['records'])}")
    print(f"\n  Output files in   : {args.output_dir}/")
    print(f"  {'─'*46}")

    all_files = sorted(f for f in os.listdir(args.output_dir)
                       if os.path.isfile(os.path.join(args.output_dir, f)))
    for fname in all_files:
        size_kb = os.path.getsize(os.path.join(args.output_dir, fname)) // 1024
        print(f"    {fname:45s}  {size_kb:6d} KB")

    print(f"\n All done!\n")


if __name__ == '__main__':
    main()