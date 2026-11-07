import argparse
from datetime import datetime
import numpy as np
import rasterio
import rasterio.windows
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
import cv2
from skimage.feature import local_binary_pattern
from collections import defaultdict, Counter
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def build_site_list_from_folders(tif_dir, shp_dir, tif_ext=".tif", shp_ext=".shp"):
    tif_dir   = Path(tif_dir)
    shp_dir   = Path(shp_dir)
    if not tif_dir.exists():
        raise FileNotFoundError(f"TIF folder tidak ditemukan: {tif_dir}")
    if not shp_dir.exists():
        raise FileNotFoundError(f"SHP folder tidak ditemukan: {shp_dir}")
    tif_files = {f.stem: f for f in sorted(tif_dir.rglob(f"*{tif_ext}"))}
    shp_files = {f.stem: f for f in sorted(shp_dir.rglob(f"*{shp_ext}"))}
    all_stems = set(tif_files) | set(shp_files)
    site_list, unmatched = [], {}
    for stem in sorted(all_stems):
        if stem in tif_files and stem in shp_files:
            site_list.append({"site": stem, "tif_path": str(tif_files[stem]), "shp_path": str(shp_files[stem])})
        elif stem in tif_files:
            unmatched[stem] = "no_shp"
        else:
            unmatched[stem] = "no_tif"
    return site_list, unmatched


def compute_vegetation_indices(r, g, b):
    r_f   = r.astype(np.float32) / 255.0
    g_f   = g.astype(np.float32) / 255.0
    b_f   = b.astype(np.float32) / 255.0
    exg   = (np.clip(2 * g_f - r_f - b_f, -1.0, 1.0) + 1.0) / 2.0
    denom = np.where(np.abs(g_f + r_f - b_f) < 1e-6, 1e-6, g_f + r_f - b_f)
    vari  = (np.clip((g_f - r_f) / denom, -1.0, 1.0) + 1.0) / 2.0
    return exg.astype(np.float32), vari.astype(np.float32)


def compute_edge(gray):
    return cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0


LBP_POINTS, LBP_RADIUS = 24, 3


def compute_texture(gray):
    # Uniform LBP codes are integers in [0, P + 1]; dividing by P + 1 gives the same
    # [0, 1] scale for every image, tile and patch (no per-image min-max).
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    return (lbp / (LBP_POINTS + 1)).astype(np.float32)


def create_7band_image(rgb_uint8):
    r, g, b   = rgb_uint8[:, :, 0], rgb_uint8[:, :, 1], rgb_uint8[:, :, 2]
    exg, vari = compute_vegetation_indices(r, g, b)
    gray      = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return np.stack([
        r.astype(np.float32) / 255.0,
        g.astype(np.float32) / 255.0,
        b.astype(np.float32) / 255.0,
        exg, vari, compute_edge(gray), compute_texture(gray),
    ], axis=-1)


def normalize_image(image, config=Config):
    mean = np.array(config.NORMALIZE_MEAN, dtype=np.float32)
    std  = np.array(config.NORMALIZE_STD,  dtype=np.float32)
    return ((image - mean) / std).astype(np.float32)


def rasterize_shapefile(shp_path, ref_tif_path, out_mask_path, kelas_col, ignore_index=255):
    gdf = gpd.read_file(shp_path)
    with rasterio.open(ref_tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        if kelas_col not in gdf.columns:
            raise ValueError(f"Kolom '{kelas_col}' tidak ada. Tersedia: {list(gdf.columns)}")
        gdf            = gdf.copy()
        gdf[kelas_col] = gdf[kelas_col].astype(int)
        shapes         = (
            (geom, int(val))
            for geom, val in zip(gdf.geometry, gdf[kelas_col])
            if geom is not None and not geom.is_empty
        )
        mask_array = rasterize(
            shapes=shapes, out_shape=(src.height, src.width),
            transform=src.transform, fill=ignore_index, dtype=np.uint8,
        )
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff", "count": 1, "dtype": "uint8", "nodata": ignore_index})
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_mask_path, "w", **out_meta) as dst:
            dst.write(mask_array, 1)
    return out_mask_path


def valid_patch_positions(mask, patch_size, stride, min_valid_ratio, ignore_index):
    H, W     = mask.shape[:2]
    total_px = patch_size * patch_size
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if ys and ys[-1] + patch_size < H: ys.append(H - patch_size)
    if xs and xs[-1] + patch_size < W: xs.append(W - patch_size)
    return [
        (y, x) for y in ys for x in xs
        if float(np.sum(mask[y:y + patch_size, x:x + patch_size] != ignore_index)) / total_px >= min_valid_ratio
    ]


def extract_patches(image, mask, patch_size, stride, min_valid_ratio, ignore_index):
    positions = valid_patch_positions(mask, patch_size, stride, min_valid_ratio, ignore_index)
    patches   = [image[y:y + patch_size, x:x + patch_size] for y, x in positions]
    mask_list = [mask[y:y + patch_size,  x:x + patch_size] for y, x in positions]
    return patches, mask_list, positions


def _collect_class_band_stats(patches_dir, masks_dir, ignore_index, n_bands=7):
    class_patch_map = defaultdict(list)
    for pf in tqdm(sorted(patches_dir.rglob("*.npy")), desc="  Collecting stats", leave=False):
        mf = masks_dir / pf.relative_to(patches_dir)
        if not mf.exists():
            continue
        patch         = np.load(pf)
        mask          = np.load(mf)
        valid_classes = np.unique(mask[mask != ignore_index])
        if len(valid_classes) == 0:
            continue
        dominant_class = max(valid_classes, key=lambda c: np.sum(mask == c))
        band_means     = patch[mask == dominant_class].mean(axis=0)
        class_patch_map[int(dominant_class)].append((pf, mf, band_means))
    return class_patch_map


def remove_outlier_patches(patches_dir, masks_dir, ignore_index=255,
                            percentile_low=5, percentile_high=95, n_bands=7):
    class_patch_map = _collect_class_band_stats(patches_dir, masks_dir, ignore_index, n_bands)
    summary = {}
    for cls_id, entries in class_patch_map.items():
        if len(entries) < 10:
            summary[cls_id] = {"total": len(entries), "removed": 0, "reason": "too_few_samples"}
            continue
        band_matrix = np.stack([e[2] for e in entries], axis=0)
        lo, hi      = np.percentile(band_matrix, percentile_low, axis=0), np.percentile(band_matrix, percentile_high, axis=0)
        removed     = 0
        for pf, mf, bm in entries:
            if np.any(bm < lo) or np.any(bm > hi):
                pf.unlink(missing_ok=True)
                mf.unlink(missing_ok=True)
                removed += 1
        summary[cls_id] = {"total": len(entries), "removed": removed, "kept": len(entries) - removed,
                           "band_lo": lo.tolist(), "band_hi": hi.tolist()}
    return summary


def balance_classes(patches_dir, masks_dir, ignore_index=255, strategy="undersample", seed=42):
    random.seed(seed)
    class_patch_map = _collect_class_band_stats(patches_dir, masks_dir, ignore_index)
    counts          = {cls: len(entries) for cls, entries in class_patch_map.items()}
    if not counts:
        return {}
    target  = int(np.median(list(counts.values())))
    summary = {"target_per_class": target, "classes": {}}
    for cls_id, entries in class_patch_map.items():
        n = len(entries)
        if strategy == "undersample" and n > target:
            for pf, mf, _ in random.sample(entries, n - target):
                pf.unlink(missing_ok=True)
                mf.unlink(missing_ok=True)
            summary["classes"][cls_id] = {"before": n, "after": target, "removed": n - target}
        else:
            summary["classes"][cls_id] = {"before": n, "after": n, "removed": 0}
    return summary


def load_patch_mask_paths(config=Config):
    all_patch_files = sorted(config.PATCHES_DIR.rglob("*.npy"))
    if not all_patch_files:
        raise FileNotFoundError(f"Tidak ada file .npy di {config.PATCHES_DIR}")
    patch_paths, mask_paths = [], []
    for pf in all_patch_files:
        mf = config.MASKS_DIR / pf.relative_to(config.PATCHES_DIR)
        if mf.exists():
            patch_paths.append(pf)
            mask_paths.append(mf)
    return patch_paths, mask_paths


def analyze_class_distribution(mask_paths, split_name, config=Config):
    class_counts = Counter()
    total_pixels = 0
    for mask_path in tqdm(mask_paths, desc=f"  Scanning {split_name:<12}", leave=False):
        mask = np.load(mask_path)
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls != config.IGNORE_INDEX:
                class_counts[cls] += count
                total_pixels      += count
    return dict(class_counts), total_pixels


def calculate_class_weights(class_counts, total_pixels, method='balanced', config=Config):
    weights = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    for cls in range(config.NUM_CLASSES):
        count = class_counts.get(cls, 0)
        if count == 0:
            continue
        if method == 'balanced':
            weights[cls] = total_pixels / (config.NUM_CLASSES * count)
        elif method == 'effective':
            beta          = 0.9999
            effective_num = 1.0 - np.power(beta, count)
            weights[cls]  = (1.0 - beta) / effective_num
        else:
            raise ValueError(f"Unknown method: {method}")
    if weights.sum() > 0:
        weights = weights / weights.sum() * config.NUM_CLASSES
    return weights.tolist()


def plot_distribution_comparison(train_counts, val_counts, train_total, val_total, config=Config):
    classes     = list(range(config.NUM_CLASSES))
    class_names = [config.CLASS_NAMES.get(c, f"C{c}") for c in classes]
    train_pct   = [(train_counts.get(c, 0) / train_total * 100) if train_total > 0 else 0 for c in classes]
    val_pct     = [(val_counts.get(c, 0)   / val_total   * 100) if val_total   > 0 else 0 for c in classes]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Train vs Validation — Class Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)

    ax1   = axes[0, 0]
    x     = np.arange(len(classes))
    width = 0.35
    bars1 = ax1.bar(x - width/2, train_pct, width, label='Train', alpha=0.85, color='steelblue', edgecolor='navy',    linewidth=1.5)
    bars2 = ax1.bar(x + width/2, val_pct,   width, label='Val',   alpha=0.85, color='coral',     edgecolor='darkred', linewidth=1.5)
    ax1.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Class Distribution Comparison (%)', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in classes], fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(max(train_pct), max(val_pct)) * 1.15)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax1.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    colors = [np.array(config.CLASS_COLORS[c]) / 255.0 for c in classes]
    for ax, counts, title in [
        (axes[0, 1], train_counts, 'Train Set Distribution'),
        (axes[1, 0], val_counts,   'Validation Set Distribution'),
    ]:
        non_zero = [(counts.get(c, 0), colors[i], class_names[i])
                    for i, c in enumerate(classes) if counts.get(c, 0) > 0]
        if non_zero:
            vals, cols, names = zip(*non_zero)
            _, _, autotexts = ax.pie(
                vals, labels=names,
                autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
                colors=cols, startangle=90,
                textprops={'fontsize': 9, 'weight': 'bold'},
            )
            for at in autotexts:
                at.set_color('white')
                at.set_fontsize(9)
        ax.set_title(title, fontweight='bold', fontsize=13)

    ax4       = axes[1, 1]
    train_log = [np.log10(train_counts.get(c, 1)) for c in classes]
    val_log   = [np.log10(val_counts.get(c, 1))   for c in classes]
    ax4.bar(x - width/2, train_log, width, label='Train', alpha=0.85, color='steelblue', edgecolor='navy',    linewidth=1.5)
    ax4.bar(x + width/2, val_log,   width, label='Val',   alpha=0.85, color='coral',     edgecolor='darkred', linewidth=1.5)
    ax4.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Log₁₀(Pixel Count)', fontweight='bold', fontsize=12)
    ax4.set_title('Class Distribution (Log Scale)', fontweight='bold', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"C{c}" for c in classes], fontsize=10)
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = config.LOG_DIR / "class_distribution_comparison.png"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return output_path


def process_single_site(site_info, config=Config):
    site_name = site_info["site"]
    tif_path  = Path(site_info["tif_path"])
    shp_path  = Path(site_info["shp_path"]) if site_info.get("shp_path") else None

    if not tif_path.exists() or shp_path is None or not shp_path.exists():
        return {"site": site_name, "n_patches": 0, "skipped": True}

    with rasterio.open(tif_path) as src:
        rgb_uint8 = src.read([1, 2, 3]).transpose(1, 2, 0)

    seven_band = create_7band_image(rgb_uint8)
    if config.NORMALIZE:
        seven_band = normalize_image(seven_band, config)

    mask_tif_path = config.MASKS_DIR / site_name / "mask.tif"
    mask_tif_path.parent.mkdir(parents=True, exist_ok=True)
    rasterize_shapefile(shp_path, tif_path, mask_tif_path, config.KELAS_COL, config.IGNORE_INDEX)

    with rasterio.open(mask_tif_path) as msrc:
        mask_array = msrc.read(1)

    patches, masks, positions = extract_patches(
        seven_band, mask_array,
        config.PATCH_SIZE, config.STRIDE, config.MIN_VALID_PIXELS, config.IGNORE_INDEX,
    )

    if not patches:
        return {"site": site_name, "n_patches": 0, "skipped": False}

    patch_out_dir = config.PATCHES_DIR / site_name
    mask_out_dir  = config.MASKS_DIR   / site_name
    patch_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for patch, mask_p, (y, x) in zip(patches, masks, positions):
        stem = f"{site_name}_y{y:05d}_x{x:05d}"
        np.save(patch_out_dir / f"{stem}.npy", patch.astype(np.dtype(config.SAVE_DTYPE_IMAGE)))
        np.save(mask_out_dir  / f"{stem}.npy", mask_p.astype(np.dtype(config.SAVE_DTYPE_MASK)))

    return {"site": site_name, "n_patches": len(patches), "skipped": False}


def _print_final_report(
    site_results, total_patches, final_count,
    outlier_summary, balance_summary,
    train_counts, val_counts, train_total, val_total,
    weights, weight_method, plot_path,
    config=Config,
):
    S  = "=" * 80
    S2 = "-" * 80

    print(f"\n{S}")
    print("📂 SITE RESULTS")
    print(S)
    for r in site_results:
        status = "SKIPPED" if r["skipped"] else f"{r['n_patches']} patches"
        print(f"  {r['site']:<30} {status}")
    print(f"  {S2}")
    print(f"  Raw extracted  : {total_patches:,}")
    print(f"  Final on disk  : {final_count:,}")

    print(f"\n{S}")
    print("🧹 OUTLIER REMOVAL")
    print(S)
    for cls_id, info in outlier_summary.items():
        name = config.CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        if info.get("reason") == "too_few_samples":
            print(f"  Class {cls_id} [{name:<18}]: skipped (< 10 samples)")
        else:
            print(f"  Class {cls_id} [{name:<18}]: removed {info['removed']:>4} / {info['total']:>4}  →  kept {info['kept']:>4}")

    if balance_summary:
        print(f"\n{S}")
        print("⚖️  CLASS BALANCING")
        print(S)
        print(f"  Target per class: {balance_summary.get('target_per_class', 'N/A')}")
        for cls_id, info in balance_summary.get("classes", {}).items():
            name = config.CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            print(f"  Class {cls_id} [{name:<18}]: {info['before']:>4} → {info['after']:>4}  (removed {info['removed']:>4})")

    print(f"\n{S}")
    print("📊 CLASS DISTRIBUTION")
    print(S)
    print(f"  {'Class':<8} {'Name':<20} {'Train px':>14}  {'Train%':>7}  {'Val px':>14}  {'Val%':>7}")
    print(f"  {S2}")
    for cls in range(config.NUM_CLASSES):
        name   = config.CLASS_NAMES.get(cls, f"Class {cls}")
        t_cnt  = train_counts.get(cls, 0)
        v_cnt  = val_counts.get(cls, 0)
        t_pct  = (t_cnt / train_total * 100) if train_total > 0 else 0
        v_pct  = (v_cnt / val_total   * 100) if val_total   > 0 else 0
        print(f"  {cls:<8} {name:<20} {t_cnt:>14,}  {t_pct:>6.2f}%  {v_cnt:>14,}  {v_pct:>6.2f}%")
    print(f"  {S2}")
    print(f"  {'TOTAL':<8} {'':<20} {train_total:>14,}  {'100.00%':>7}  {val_total:>14,}  {'100.00%':>7}")

    print(f"\n{S}")
    print(f"⚖️  CLASS WEIGHTS  ({weight_method.upper()})")
    print(S)
    for cls in range(config.NUM_CLASSES):
        name  = config.CLASS_NAMES.get(cls, f"Class {cls}")
        count = train_counts.get(cls, 0)
        ratio = (train_total / count) if count > 0 else 0
        tag   = f"1:{ratio:.1f}" if count > 0 else "NO SAMPLES"
        print(f"  Class {cls} [{name:<18}]: {weights[cls]:>8.4f}  ({tag})")
    print(f"  {S2}")
    print(f"  CLASS_WEIGHTS = {weights}")
    max_ratio = max((train_total / c if c > 0 else 0 for c in train_counts.values()), default=0)
    if max_ratio < 10:
        print(f"\n  ✅ Imbalance ringan  (< 10:1)   → CLASS_WEIGHTS = None")
    elif max_ratio < 100:
        print(f"\n  ⚠️  Imbalance sedang  (10–100:1) → Gunakan weights di atas")
    else:
        print(f"\n  🚨 Imbalance ekstrem (> 100:1)  → Weights + FOCAL_GAMMA = 3.0")

    if getattr(config, 'CONFUSING_PAIRS', None):
        print(f"\n{S}")
        print("🔍 CONFUSING PAIRS")
        print(S)
        for cls1, cls2 in config.CONFUSING_PAIRS:
            n1     = config.CLASS_NAMES.get(cls1, f"Class {cls1}")
            n2     = config.CLASS_NAMES.get(cls2, f"Class {cls2}")
            t1, t2 = train_counts.get(cls1, 0), train_counts.get(cls2, 0)
            v1, v2 = val_counts.get(cls1, 0),   val_counts.get(cls2, 0)
            print(f"  {n1} (C{cls1}) ↔ {n2} (C{cls2})")
            print(f"    Train: {t1:>12,} vs {t2:>12,}")
            print(f"    Val  : {v1:>12,} vs {v2:>12,}")
            if t1 > 0 and t2 > 0:
                ratio = max(t1, t2) / min(t1, t2)
                flag  = "⚠️  High" if ratio > 10 else ("⚠️  Moderate" if ratio > 5 else "✅ OK")
                print(f"    Ratio: {ratio:.1f}x  {flag}")
            else:
                print("    ❌ One or both classes have 0 pixels!")

    tv = [c for c in train_counts.values() if c > 0]
    if tv:
        imb = max(tv) / min(tv)
        missing = set(train_counts) - set(val_counts)
        max_diff = max(
            abs((train_counts.get(c, 0) / train_total * 100 if train_total else 0) -
                (val_counts.get(c, 0)   / val_total   * 100 if val_total   else 0))
            for c in range(config.NUM_CLASSES)
        )
        print(f"\n{S}")
        print("📈 SUMMARY")
        print(S)
        print(f"  Train imbalance ratio : {imb:.1f}x")
        print(f"  Max train/val pct diff: {max_diff:.2f}%")
        print(f"  Missing cls in val    : {sorted(missing) if missing else 'None'}")
        if plot_path:
            print(f"  Plot saved            : {plot_path}")

    print(f"\n{S}")
    print("✅ PIPELINE COMPLETE")
    print(S + "\n")


def split_blocks(height, width, block_size):
    blocks = []
    for y0 in range(0, height, block_size):
        for x0 in range(0, width, block_size):
            blocks.append((y0, x0, min(block_size, height - y0), min(block_size, width - x0)))
    return blocks


def scan_site_blocks(site_info, config=Config):
    """Rasterise the labels and list, per spatial block, the class histogram and the
    patch windows that fit entirely inside the block. No image features needed yet."""
    site_name = site_info["site"]
    tif_path  = Path(site_info["tif_path"])
    shp_path  = Path(site_info["shp_path"])

    mask_tif_path = config.LABEL_RASTER_DIR / f"{site_name}.tif"
    rasterize_shapefile(shp_path, tif_path, mask_tif_path, config.KELAS_COL, config.IGNORE_INDEX)
    with rasterio.open(mask_tif_path) as msrc:
        mask_array = msrc.read(1)
        transform  = msrc.transform
        crs        = msrc.crs

    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf[config.KELAS_COL] = gdf[config.KELAS_COL].astype(int)
    polygon_stats = {
        int(c): {"polygons": int(len(g)), "area_ha": float(g.geometry.area.sum() / 1e4)}
        for c, g in gdf.groupby(config.KELAS_COL)
    }

    H, W   = mask_array.shape
    blocks = []
    for bi, (y0, x0, bh, bw) in enumerate(split_blocks(H, W, config.SPLIT_BLOCK_SIZE)):
        mblock    = mask_array[y0:y0 + bh, x0:x0 + bw]
        positions = valid_patch_positions(mblock, config.PATCH_SIZE, config.STRIDE,
                                          config.MIN_VALID_PIXELS, config.IGNORE_INDEX)
        if not positions:
            continue
        hist = np.bincount(mblock[mblock != config.IGNORE_INDEX].ravel(),
                           minlength=config.NUM_CLASSES)[:config.NUM_CLASSES]
        blocks.append({
            "site": site_name, "block_id": f"{site_name}_b{bi:04d}",
            "y0": y0, "x0": x0, "h": bh, "w": bw,
            "bounds": list(rasterio.windows.bounds(rasterio.windows.Window(x0, y0, bw, bh), transform)),
            "hist": hist.astype(np.int64),
            "positions": [(y0 + y, x0 + x) for y, x in positions],
        })
    return {"site": site_name, "shape": (H, W), "crs": str(crs), "transform": transform,
            "mask_path": mask_tif_path, "polygon_stats": polygon_stats, "blocks": blocks}


def assign_blocks_to_splits(blocks, fractions, seed, iters):
    """Random search over block permutations: blocks are cut into consecutive runs
    whose labelled-pixel shares match `fractions`; keep the draw whose per-split
    class mix is closest to the overall mix (every class must occur in every split)."""
    names   = list(fractions)
    targets = np.array([fractions[n] for n in names], dtype=float)
    cuts    = np.cumsum(targets)[:-1]
    hist    = np.stack([b["hist"] for b in blocks]).astype(float)
    weight  = hist.sum(1)
    total   = weight.sum()
    present = hist.sum(0) > 0
    overall = hist.sum(0)[present] / total

    rng = np.random.default_rng(seed)
    best_score, best_assign = np.inf, None
    for _ in range(iters):
        order  = rng.permutation(len(blocks))
        mid    = (np.cumsum(weight[order]) - 0.5 * weight[order]) / total
        assign = np.empty(len(blocks), dtype=int)
        assign[order] = np.searchsorted(cuts, mid, side="right")

        score = 0.0
        for k in range(len(names)):
            h_k = hist[assign == k].sum(0)[present]
            if h_k.sum() == 0 or (h_k == 0).any():
                score = np.inf
                break
            score += abs(h_k.sum() / total - targets[k])
            score += np.abs(h_k / h_k.sum() - overall).max()
        if score < best_score:
            best_score, best_assign = score, assign.copy()

    if best_assign is None:
        raise RuntimeError("No block assignment puts every class in every split.")
    for b, k in zip(blocks, best_assign):
        b["split"] = names[k]
    return float(best_score)


def extract_site_patches(site_info, site_scan, config=Config):
    """Compute the 7-band stack once per orthomosaic, then save every patch window
    of every assigned block into PATCHES_DIR/<split>/<site>/."""
    site_name = site_info["site"]
    with rasterio.open(site_info["tif_path"]) as src:
        rgb_uint8 = src.read([1, 2, 3]).transpose(1, 2, 0)
    seven_band = create_7band_image(rgb_uint8)
    del rgb_uint8
    if config.NORMALIZE:
        # In place: a second full-size float32 copy of a large orthomosaic does not fit in RAM.
        seven_band -= np.array(config.NORMALIZE_MEAN, dtype=np.float32)
        seven_band /= np.array(config.NORMALIZE_STD,  dtype=np.float32)
    with rasterio.open(site_scan["mask_path"]) as msrc:
        mask_array = msrc.read(1)

    records = []
    P = config.PATCH_SIZE
    for block in site_scan["blocks"]:
        split = block["split"]
        patch_out_dir = config.PATCHES_DIR / split / site_name
        mask_out_dir  = config.MASKS_DIR   / split / site_name
        patch_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        for (y, x) in block["positions"]:
            stem   = f"{site_name}_y{y:05d}_x{x:05d}"
            mask_p = mask_array[y:y + P, x:x + P]
            np.save(patch_out_dir / f"{stem}.npy",
                    seven_band[y:y + P, x:x + P].astype(np.dtype(config.SAVE_DTYPE_IMAGE)))
            np.save(mask_out_dir / f"{stem}.npy", mask_p.astype(np.dtype(config.SAVE_DTYPE_MASK)))
            labelled = mask_p[mask_p != config.IGNORE_INDEX]
            records.append({
                "patch_id": stem, "site": site_name, "block_id": block["block_id"], "split": split,
                "row": y, "col": x,
                "patch": str((patch_out_dir / f"{stem}.npy").relative_to(config.PROCESSED_DATA_DIR)),
                "mask":  str((mask_out_dir  / f"{stem}.npy").relative_to(config.PROCESSED_DATA_DIR)),
                "labelled_px": int(labelled.size),
                "dominant_class": int(np.bincount(labelled).argmax()),
            })
    return records


def verify_no_overlap(site_scan, records, patch_size):
    """Pixel-level check that no pixel is covered by patches of two different splits."""
    H, W  = site_scan["shape"]
    bits  = {"train": 1, "val": 2, "test": 4}
    cover = np.zeros((H, W), dtype=np.uint8)
    for r in records:
        if r["site"] == site_scan["site"]:
            cover[r["row"]:r["row"] + patch_size, r["col"]:r["col"] + patch_size] |= bits[r["split"]]
    shared = int(np.count_nonzero(cover & (cover - 1)))
    if shared:
        raise RuntimeError(f"{site_scan['site']}: {shared} pixels are shared between splits")
    return shared


def write_split_layers(site_scans, records, config=Config):
    import pandas as pd
    from shapely.geometry import box

    block_rows = []
    for scan in site_scans:
        for b in scan["blocks"]:
            kept = sum(1 for r in records if r["block_id"] == b["block_id"] and r["status"] == "kept")
            row  = {"site": b["site"], "block_id": b["block_id"], "split": b["split"],
                    "y0": b["y0"], "x0": b["x0"], "h": b["h"], "w": b["w"],
                    "labelled_px": int(b["hist"].sum()), "patches_extracted": len(b["positions"]),
                    "patches_kept": kept, "crs": scan["crs"], "geometry": box(*b["bounds"])}
            row.update({f"px_c{c}": int(b["hist"][c]) for c in config.EVAL_CLASSES})
            block_rows.append(row)
    blocks_gdf = pd.DataFrame(block_rows)
    for crs, part in blocks_gdf.groupby("crs"):
        gpd.GeoDataFrame(part.drop(columns="crs"), geometry="geometry", crs=crs).to_file(
            config.PROCESSED_DATA_DIR / "split_blocks.gpkg", layer=f"blocks_{crs.replace(':', '')}")

    pd.DataFrame(records).to_csv(config.PROCESSED_DATA_DIR / "split_manifest.csv", index=False)


def run_spatial_preprocessing(
    site_list,
    config           = Config,
    outlier_low      = 5,
    outlier_high     = 95,
    balance_strategy = "undersample",
    weight_method    = "balanced",
    visualize        = True,
):
    """Leak-free order: blocks -> split -> patches -> outlier removal and balancing on
    the training split only. Validation and test keep their natural distribution."""
    import json
    import time

    config.make_dirs()
    config.LABEL_RASTER_DIR.mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.perf_counter()
    log.info("STEP 1/6 — Rasterise labels and scan %d px blocks", config.SPLIT_BLOCK_SIZE)
    site_scans = [scan_site_blocks(s, config) for s in tqdm(site_list, desc="Scanning sites")]
    all_blocks = [b for scan in site_scans for b in scan["blocks"]]
    timings["rasterise_and_scan_s"] = time.perf_counter() - t0

    log.info("STEP 2/6 — Assign %d blocks to splits %s", len(all_blocks), config.SPLIT_FRACTIONS)
    split_score = assign_blocks_to_splits(all_blocks, config.SPLIT_FRACTIONS,
                                          config.SPLIT_SEED, config.SPLIT_SEARCH_ITERS)

    t0 = time.perf_counter()
    log.info("STEP 3/6 — Feature stack and patch extraction")
    records = []
    for site_info, scan in zip(site_list, site_scans):
        records += extract_site_patches(site_info, scan, config)
    timings["features_and_patches_s"] = time.perf_counter() - t0
    for scan in site_scans:
        verify_no_overlap(scan, records, config.PATCH_SIZE)
    log.info("  Overlap check passed: no pixel is shared between train, val and test")

    train_patch_dir = config.PATCHES_DIR / "train"
    train_mask_dir  = config.MASKS_DIR / "train"

    t0 = time.perf_counter()
    outlier_summary = {}
    if outlier_low is not None:
        log.info("STEP 4/6 — Outlier removal (training split only)")
        outlier_summary = remove_outlier_patches(train_patch_dir, train_mask_dir, config.IGNORE_INDEX,
                                                 outlier_low, outlier_high)
    after_outlier = {p.stem for p in train_patch_dir.rglob("*.npy")}

    balance_summary = {}
    if balance_strategy:
        log.info("STEP 5/6 — Class balancing (training split only, %s)", balance_strategy)
        balance_summary = balance_classes(train_patch_dir, train_mask_dir, config.IGNORE_INDEX,
                                          strategy=balance_strategy, seed=config.SPLIT_SEED)
    after_balance = {p.stem for p in train_patch_dir.rglob("*.npy")}
    timings["filter_and_balance_s"] = time.perf_counter() - t0

    for r in records:
        if r["split"] != "train" or r["patch_id"] in after_balance:
            r["status"] = "kept"
        elif r["patch_id"] in after_outlier:
            r["status"] = "removed_balance"
        else:
            r["status"] = "removed_outlier"

    log.info("STEP 6/6 — Distribution, class weights and manifest")
    kept = {s: [r for r in records if r["split"] == s and r["status"] == "kept"]
            for s in config.SPLIT_FRACTIONS}
    counts, totals = {}, {}
    for s, recs in kept.items():
        counts[s], totals[s] = analyze_class_distribution(
            [config.PROCESSED_DATA_DIR / r["mask"] for r in recs], s, config)
        counts[s] = {int(k): int(v) for k, v in counts[s].items()}
    weights = calculate_class_weights(counts["train"], totals["train"], method=weight_method, config=config)

    storage_bytes = sum((config.PROCESSED_DATA_DIR / r["patch"]).stat().st_size
                        for recs in kept.values() for r in recs)
    write_split_layers(site_scans, records, config)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "sites": [{"site": s["site"], "tif": si["tif_path"], "shp": si["shp_path"],
                   "shape_px": list(s["shape"]), "crs": s["crs"],
                   "polygons_per_class": s["polygon_stats"]}
                  for s, si in zip(site_scans, site_list)],
        "settings": {
            "block_size_px": config.SPLIT_BLOCK_SIZE, "fractions": config.SPLIT_FRACTIONS,
            "split_seed": config.SPLIT_SEED, "search_iters": config.SPLIT_SEARCH_ITERS,
            "patch_size": config.PATCH_SIZE, "stride": config.STRIDE,
            "min_valid_ratio": config.MIN_VALID_PIXELS,
            "outlier_percentiles": None if outlier_low is None else [outlier_low, outlier_high],
            "balance_strategy": balance_strategy, "weight_method": weight_method,
            "band_names": config.BAND_NAMES, "normalize_mean": config.NORMALIZE_MEAN,
            "normalize_std": config.NORMALIZE_STD,
        },
        "split_score": split_score,
        "blocks": {s: sum(1 for b in all_blocks if b["split"] == s) for s in config.SPLIT_FRACTIONS},
        "patches": {s: [{"patch": r["patch"], "mask": r["mask"]} for r in recs] for s, recs in kept.items()},
        "patch_counts": {
            s: {st: sum(1 for r in records if r["split"] == s and r["status"] == st)
                for st in ("kept", "removed_outlier", "removed_balance")}
            for s in config.SPLIT_FRACTIONS},
        "class_pixels": counts,
        "class_weights": weights,
        "outlier_summary": {int(k): v for k, v in outlier_summary.items()},
        "balance_summary": {**balance_summary,
                            "classes": {int(k): v for k, v in balance_summary.get("classes", {}).items()}},
        "timings_s": timings,
        "storage_bytes_kept": storage_bytes,
    }
    with open(config.SPLIT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else int(o))

    plot_path = None
    if visualize:
        try:
            plot_path = plot_distribution_comparison(counts["train"], counts["val"],
                                                     totals["train"], totals["val"], config)
        except Exception as e:
            log.warning("Plot error: %s", e)

    site_results = [{"site": s["site"], "n_patches": sum(len(b["positions"]) for b in s["blocks"]),
                     "skipped": False} for s in site_scans]
    _print_final_report(
        site_results, len(records), sum(len(v) for v in kept.values()),
        outlier_summary, balance_summary,
        counts["train"], counts["val"], totals["train"], totals["val"],
        weights, weight_method, plot_path, config,
    )
    for s in config.SPLIT_FRACTIONS:
        log.info("  %-5s blocks %3d | patches kept %5d %s", s, manifest["blocks"][s],
                 len(kept[s]), manifest["patch_counts"][s])
    log.info("  Manifest: %s", config.SPLIT_MANIFEST)
    return manifest


def run_preprocessing(
    site_list,
    config           = Config,
    outlier_low      = 5,
    outlier_high     = 95,
    balance_strategy = "undersample",
    weight_method    = "balanced",
    analyze          = True,
    visualize        = True,
):
    config.make_dirs()

    log.info("STEP 1/5 — Patch extraction (%d sites)", len(site_list))
    site_results, total_patches = [], 0
    for site_info in tqdm(site_list, desc="Extracting patches"):
        result         = process_single_site(site_info, config)
        site_results.append(result)
        total_patches += result.get("n_patches", 0)

    log.info("STEP 2/5 — Outlier removal")
    outlier_summary = remove_outlier_patches(
        config.PATCHES_DIR, config.MASKS_DIR,
        ignore_index=config.IGNORE_INDEX,
        percentile_low=outlier_low, percentile_high=outlier_high,
    )

    balance_summary = {}
    if balance_strategy:
        log.info("STEP 3/5 — Class balancing (%s)", balance_strategy)
        balance_summary = balance_classes(
            config.PATCHES_DIR, config.MASKS_DIR,
            ignore_index=config.IGNORE_INDEX, strategy=balance_strategy,
        )

    final_count = sum(1 for _ in config.PATCHES_DIR.rglob("*.npy"))

    train_counts, val_counts   = {}, {}
    train_total,  val_total    = 0, 0
    weights                    = [1.0] * config.NUM_CLASSES
    plot_path                  = None

    if analyze:
        log.info("STEP 4/5 — Distribution analysis")
        patch_paths, mask_paths = load_patch_mask_paths(config)
        train_p, val_p, train_m, val_m = train_test_split(
            patch_paths, mask_paths,
            test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED, shuffle=True,
        )
        train_counts, train_total = analyze_class_distribution(train_m, "train", config)
        val_counts,   val_total   = analyze_class_distribution(val_m,   "val",   config)
        weights                   = calculate_class_weights(train_counts, train_total, method=weight_method, config=config)

        if visualize:
            log.info("STEP 5/5 — Visualization")
            try:
                plot_path = plot_distribution_comparison(train_counts, val_counts, train_total, val_total, config)
            except Exception as e:
                log.warning("Plot error: %s", e)

    _print_final_report(
        site_results, total_patches, final_count,
        outlier_summary, balance_summary,
        train_counts, val_counts, train_total, val_total,
        weights, weight_method, plot_path,
        config,
    )

    return {
        "patches_dir"         : str(config.PATCHES_DIR),
        "masks_dir"           : str(config.MASKS_DIR),
        "site_results"        : site_results,
        "total_raw_patches"   : total_patches,
        "total_final_patches" : final_count,
        "outlier_summary"     : outlier_summary,
        "balance_summary"     : balance_summary,
        "train_counts"        : train_counts,
        "val_counts"          : val_counts,
        "train_total"         : train_total,
        "val_total"           : val_total,
        "class_weights"       : weights,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build segmentation training dataset from TIF + SHP folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tif_dir",          type=str, required=True)
    parser.add_argument("--shp_dir",          type=str, required=True)
    parser.add_argument("--tif_ext",          type=str, default=".tif")
    parser.add_argument("--shp_ext",          type=str, default=".shp")
    parser.add_argument("--weight_method",    type=str, default="balanced", choices=["balanced", "effective"])
    parser.add_argument("--balance_strategy", type=str, default="undersample", choices=["undersample", "none"])
    parser.add_argument("--outlier_low",      type=int, default=5)
    parser.add_argument("--outlier_high",     type=int, default=95)
    parser.add_argument("--no_outlier_removal", action="store_true",
                        help="Skip the per-class P5-P95 band-mean filter (it selects patches using the engineered bands).")
    parser.add_argument("--legacy_random_split", action="store_true",
                        help="Original pipeline: filter and balance all patches, then split at random (leaks).")
    parser.add_argument("--no_analyze",       action="store_true")
    parser.add_argument("--no_visualize",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    site_list, unmatched = build_site_list_from_folders(
        tif_dir=args.tif_dir, shp_dir=args.shp_dir,
        tif_ext=args.tif_ext, shp_ext=args.shp_ext,
    )

    if unmatched:
        for stem, reason in unmatched.items():
            log.warning("Unmatched [%s]: %s", stem, reason)

    if not site_list:
        log.error("Tidak ada site yang bisa diproses. Periksa folder TIF dan SHP.")
        exit(1)

    log.info("Found %d paired sites", len(site_list))

    if args.legacy_random_split:
        run_preprocessing(
            site_list,
            config           = Config,
            outlier_low      = args.outlier_low,
            outlier_high     = args.outlier_high,
            balance_strategy = None if args.balance_strategy == "none" else args.balance_strategy,
            weight_method    = args.weight_method,
            analyze          = not args.no_analyze,
            visualize        = not args.no_visualize,
        )
    else:
        run_spatial_preprocessing(
            site_list,
            config           = Config,
            outlier_low      = None if args.no_outlier_removal else args.outlier_low,
            outlier_high     = None if args.no_outlier_removal else args.outlier_high,
            balance_strategy = None if args.balance_strategy == "none" else args.balance_strategy,
            weight_method    = args.weight_method,
            visualize        = not args.no_visualize,
        )
