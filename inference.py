"""
pipeline.py - Complete Pipeline: Resample → Inference → QGIS Style
Usage:
    python pipeline.py --input image.tif --resolution 0.3
    python pipeline.py --input folder/ --resolution 0.3 --model best_model.pth
"""

import argparse
import logging
import os
import platform
import random
import sys
import warnings
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import reproject
from shapely.geometry import shape as shapely_shape
import torch
import torch.nn.functional as F
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom

from config import Config
from build_training_dataset import create_7band_image, normalize_image
from train import SegFormer7Band
from filter_small_pixels import filter_small_pixels
from smoothing_topojson import smooth_polygons_topojson

try:
    from torch.amp import autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast
    _AMP_DEVICE = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# GLOBAL CONTEXT
# ============================================================================

def setup_pipeline_logging(config=Config) -> logging.Logger:
    """
    Centralised logger shared across all pipeline stages.
    Writes to stdout (INFO+) and a timestamped log file (DEBUG+).
    Re-uses existing handlers if the root logger is already configured
    (e.g. when pipeline.py is imported alongside train.py).
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Avoid duplicate handlers when module is re-imported
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and
               not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        root.addHandler(sh)

    root.setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline log : {log_file}")
    return logger


def set_pipeline_seed(seed: int) -> None:
    """Fix all random seeds for reproducible inference results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


def resolve_pipeline_device(config=Config) -> torch.device:
    """Resolve compute device and log hardware details."""
    logger = logging.getLogger(__name__)

    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        logger.info(f"Device     : CUDA — {props.name}")
        logger.info(f"  VRAM     : {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  SM count : {props.multi_processor_count}")
        logger.info(f"  CUDA ver : {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        if config.DEVICE == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU.")
        logger.info(f"Device     : CPU ({torch.get_num_threads()} threads)")

    return device


def log_pipeline_environment(config=Config) -> None:
    """Dump full environment snapshot for reproducibility audits."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("ENVIRONMENT SNAPSHOT")
    logger.info("=" * 60)
    logger.info(f"  Python    : {sys.version.split()[0]}")
    logger.info(f"  Platform  : {platform.platform()}")
    logger.info(f"  PyTorch   : {torch.__version__}")
    logger.info(f"  CUDA avail: {torch.cuda.is_available()}")
    logger.info(f"  CuPy avail: {CUPY_AVAILABLE}")

    for pkg in ("rasterio", "geopandas", "shapely", "cv2", "transformers"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            logger.info(f"  {pkg:<12}: {ver}")
        except ImportError:
            pass

    logger.info(f"  NumPy     : {np.__version__}")
    logger.info("=" * 60)


def log_pipeline_config(config=Config) -> None:
    """Log all public Config attributes relevant to inference."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("CONFIG SUMMARY (pipeline)")
    logger.info("=" * 60)
    for key in dir(config):
        if key.startswith("_"):
            continue
        val = getattr(config, key)
        if callable(val):
            continue
        logger.info(f"  {key:<30}: {val}")
    logger.info("=" * 60)


def log_job_summary(tif_files: list, job_dir: Path, model_path: Path,
                    resolution: Optional[float], config=Config) -> None:
    """Log job-level summary before processing begins."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("JOB SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Files to process : {len(tif_files)}")
    logger.info(f"  Job directory    : {job_dir}")
    logger.info(f"  Model path       : {model_path}")
    logger.info(f"  Target resolution: {resolution if resolution else 'original'}")
    logger.info(f"  Num classes      : {config.NUM_CLASSES}")
    logger.info(f"  Ignore index     : {config.IGNORE_INDEX}")
    logger.info("=" * 60)


def log_model_info(model: torch.nn.Module) -> None:
    """Log total / trainable / frozen parameter counts."""
    logger = logging.getLogger(__name__)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 60)
    logger.info("MODEL INFO")
    logger.info("=" * 60)
    logger.info(f"  Total params    : {total:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    logger.info(f"  Frozen params   : {total - trainable:,}")
    logger.info("=" * 60)


class PipelineContext:
    """
    Lightweight singleton that bootstraps and carries all shared
    pipeline runtime state: logger, device, seed, start time.

    Mirrors GlobalContext in train.py so both can coexist without
    double-initialising handlers when used together.
    """
    _instance = None

    def __new__(cls, config=Config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(config)
        return cls._instance

    def _init(self, config):
        self.config     = config
        self.logger     = setup_pipeline_logging(config)
        set_pipeline_seed(config.RANDOM_SEED)
        self.device     = resolve_pipeline_device(config)
        log_pipeline_environment(config)
        log_pipeline_config(config)
        self.logger.info(f"Pipeline seed set to {config.RANDOM_SEED}")
        self.start_time = datetime.now()

    def elapsed(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def finalize(self, result: dict) -> None:
        elapsed_min = self.elapsed() / 60
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info(f"  Elapsed      : {elapsed_min:.1f} min")
        self.logger.info(f"  Job dir      : {result.get('job_dir', '—')}")
        self.logger.info(f"  Files        : {result.get('success', 0)}/{result.get('total', 0)}")
        self.logger.info(f"  Total time   : {result.get('time', '—')}")
        failed = [r for r in result.get('results', []) if 'error' in r]
        if failed:
            self.logger.warning(f"  Failed files : {len(failed)}")
            for r in failed:
                self.logger.warning(f"    ✗ {r['input']} — {r['error']}")
        self.logger.info("=" * 60)


# ============================================================================
# RESAMPLE FUNCTIONS
# ============================================================================

def resample_tif(input_tif: Path, output_tif: Path, target_resolution: float,
                 resampling_method: Resampling = Resampling.bilinear) -> dict:
    """Resample TIF to target resolution."""
    logger = logging.getLogger(__name__)
    logger.info(f"Resampling {input_tif.name} → {target_resolution}m ...")

    with rasterio.open(input_tif) as src:
        src_transform = src.transform
        src_res  = (abs(src_transform.a) + abs(src_transform.e)) / 2
        scale    = src_res / target_resolution
        new_w    = int(src.width  * scale)
        new_h    = int(src.height * scale)
        new_transform = rasterio.Affine(
            target_resolution, 0.0, src_transform.c,
            0.0, -target_resolution, src_transform.f,
        )

        out_profile = src.profile.copy()
        out_profile.update({
            'width': new_w, 'height': new_h, 'transform': new_transform,
            'compress': 'lzw', 'tiled': True,
            'blockxsize': 512, 'blockysize': 512,
        })

        output_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_tif, 'w', **out_profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst_band = np.empty((new_h, new_w), dtype=src.dtypes[0])
                reproject(
                    source=src.read(band_idx), destination=dst_band,
                    src_transform=src_transform, src_crs=src.crs,
                    dst_transform=new_transform, dst_crs=src.crs,
                    resampling=resampling_method,
                    src_nodata=src.nodata, dst_nodata=src.nodata,
                )
                dst.write(dst_band, band_idx)

    info = {'output': str(output_tif), 'dimensions': f"{new_w}x{new_h}"}
    logger.info(f"  Resampled → {info['dimensions']} px")
    return info


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def make_gaussian_weight(size: int, sigma_ratio: float = 6.0) -> np.ndarray:
    """Create Gaussian weight map."""
    sigma  = size / sigma_ratio
    center = size / 2.0 - 0.5
    y, x   = np.meshgrid(
        np.arange(size) - center,
        np.arange(size) - center,
        indexing='ij',
    )
    weight = np.exp(-(y**2 + x**2) / (2 * sigma**2))
    return (weight / (weight.max() + 1e-9)).astype(np.float32)


def generate_patches(H: int, W: int, patch_size: int,
                     overlap: int) -> List[Tuple[int, int, int, int]]:
    """Generate patch positions."""
    stride = max(patch_size - overlap, 1)
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if not ys or ys[-1] + patch_size < H:
        ys.append(max(0, H - patch_size))
    if not xs or xs[-1] + patch_size < W:
        xs.append(max(0, W - patch_size))

    patches = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            patches.append((y, x, min(y + patch_size, H), min(x + patch_size, W)))
    return patches


def run_inference_on_tile(seven_band: np.ndarray, model: torch.nn.Module,
                          patch_size: int = 512, batch_size: int = 4,
                          sigma_ratio: float = 6.0, config=Config) -> np.ndarray:
    """Run inference on a single tile with Gaussian-weighted patch blending."""
    logger   = logging.getLogger(__name__)
    H, W, C  = seven_band.shape
    device   = next(model.parameters()).device
    n_cls    = config.NUM_CLASSES
    overlap  = int(patch_size * 0.25)

    logit_acc  = np.zeros((n_cls, H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W),        dtype=np.float32)
    gauss_w    = make_gaussian_weight(patch_size, sigma_ratio)

    patch_positions = generate_patches(H, W, patch_size, overlap)
    img_chw         = seven_band.transpose(2, 0, 1)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(patch_positions), batch_size):
            batch_pos     = patch_positions[batch_start:batch_start + batch_size]
            batch_tensors = []
            batch_meta    = []

            for (y, x, y_end, x_end) in batch_pos:
                actual_h, actual_w = y_end - y, x_end - x
                patch = img_chw[:, y:y_end, x:x_end]
                if actual_h < patch_size or actual_w < patch_size:
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, patch_size - actual_h), (0, patch_size - actual_w)),
                        mode='reflect',
                    )
                batch_tensors.append(patch)
                batch_meta.append((y, x, actual_h, actual_w))

            batch_t = torch.from_numpy(np.stack(batch_tensors)).float().to(device)

            if _AMP_DEVICE and device.type == 'cuda':
                with autocast(device_type=_AMP_DEVICE):
                    logits = model(batch_t)
            else:
                logits = model(batch_t)

            probs = F.softmax(logits, dim=1).cpu().numpy()

            for i, (y, x, actual_h, actual_w) in enumerate(batch_meta):
                prob_patch = probs[i, :, :actual_h, :actual_w]
                w_patch    = gauss_w[:actual_h, :actual_w]
                logit_acc[:, y:y + actual_h, x:x + actual_w] += prob_patch * w_patch[np.newaxis]
                weight_acc[y:y + actual_h, x:x + actual_w]   += w_patch

    weight_acc = np.maximum(weight_acc, 1e-9)
    return (logit_acc / weight_acc[np.newaxis]).argmax(axis=0).astype(np.uint8)


def cuda_tiling_inference(tif_path: Path, model: torch.nn.Module,
                          tile_size: int = 2048, tile_overlap: int = 64,
                          patch_size: int = 512, patch_batch_size: int = 4,
                          sigma_ratio: float = 6.0,
                          config=Config) -> Tuple[np.ndarray, dict]:
    """CUDA tiling inference with Gaussian-weighted tile blending."""
    logger = logging.getLogger(__name__)

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        H, W    = src.height, src.width
        logger.info(f"  Image size   : {W}×{H} px")

        stride   = max(tile_size - tile_overlap, 1)
        tile_ys  = list(range(0, H, stride))
        tile_xs  = list(range(0, W, stride))
        if tile_ys[-1] + tile_size < H:
            tile_ys.append(max(0, H - tile_size))
        if tile_xs[-1] + tile_size < W:
            tile_xs.append(max(0, W - tile_size))

        tile_positions = []
        for ty in sorted(set(tile_ys)):
            for tx in sorted(set(tile_xs)):
                tile_positions.append((
                    ty, tx,
                    min(ty + tile_size, H) - ty,
                    min(tx + tile_size, W) - tx,
                ))

        logger.info(f"  Tile count   : {len(tile_positions)}")

        pred_acc_gpu    = cp.zeros((config.NUM_CLASSES, H, W), dtype=cp.float32)
        weight_acc_gpu  = cp.zeros((H, W), dtype=cp.float32)
        tile_weight_gpu = cp.asarray(make_gaussian_weight(tile_size, sigma_ratio=4.0))

        model.eval()
        with torch.no_grad():
            for (ty, tx, tile_h, tile_w) in tqdm(
                    tile_positions, desc="CUDA Tiling", leave=False):

                window   = rasterio.windows.Window(tx, ty, tile_w, tile_h)
                rgb_tile = src.read([1, 2, 3], window=window)

                rgb_uint8 = np.zeros_like(rgb_tile, dtype=np.uint8)
                for b in range(3):
                    band = rgb_tile[b].astype(np.float32)
                    mn, mx = np.nanmin(band), np.nanmax(band)
                    if mx > mn:
                        band = (band - mn) / (mx - mn)
                    rgb_uint8[b] = np.clip(band * 255, 0, 255).astype(np.uint8)

                seven_band_tile = create_7band_image(rgb_uint8.transpose(1, 2, 0))
                if config.NORMALIZE:
                    seven_band_tile = normalize_image(seven_band_tile, config)

                pred_tile     = run_inference_on_tile(
                    seven_band_tile, model, patch_size,
                    patch_batch_size, sigma_ratio, config,
                )
                pred_tile_gpu = cp.asarray(pred_tile)

                pred_onehot_gpu = cp.zeros(
                    (config.NUM_CLASSES, tile_h, tile_w), dtype=cp.float32)
                for c in range(config.NUM_CLASSES):
                    pred_onehot_gpu[c] = (pred_tile_gpu == c).astype(cp.float32)

                tw_gpu = tile_weight_gpu[:tile_h, :tile_w]
                pred_acc_gpu[:, ty:ty+tile_h, tx:tx+tile_w] += \
                    pred_onehot_gpu * tw_gpu[cp.newaxis]
                weight_acc_gpu[ty:ty+tile_h, tx:tx+tile_w] += tw_gpu

        weight_acc_gpu = cp.maximum(weight_acc_gpu, 1e-9)
        pred_map = cp.asnumpy(
            cp.argmax(pred_acc_gpu / weight_acc_gpu[cp.newaxis], axis=0)
            .astype(cp.uint8)
        )

    return pred_map, profile


def raster_to_shp(pred_map: np.ndarray, profile: dict, out_shp_path: Path,
                  class_field: str = "kelas", config=Config) -> Path:
    """Convert raster prediction map to shapefile."""
    logger = logging.getLogger(__name__)

    records = []
    for geom_mapping, value in rasterio_shapes(
            pred_map,
            mask=(pred_map != config.IGNORE_INDEX).astype(np.uint8),
            transform=profile['transform']):
        if int(value) != config.IGNORE_INDEX:
            records.append({
                'geometry'  : shapely_shape(geom_mapping),
                class_field : int(value),
            })

    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=profile['crs'])
    gdf = (gdf.dissolve(by=class_field, as_index=False)
              .explode(index_parts=False)
              .reset_index(drop=True))

    out_shp_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_shp_path))
    logger.debug(f"  Raw shapefile: {out_shp_path} ({len(gdf)} polygons)")
    return out_shp_path


# ============================================================================
# QGIS STYLE FUNCTIONS
# ============================================================================

def create_qml_style(class_colors: dict, class_names: dict, class_field: str,
                     output_qml: Path, alpha: int = 200) -> Path:
    """Create QGIS QML style file."""
    logger           = logging.getLogger(__name__)
    alpha_normalized = alpha / 255.0

    qml_content = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="{class_field}" symbollevels="0" forceraster="0" enableorderby="0">
    <categories>
'''
    for class_id in sorted(class_colors.keys()):
        class_name = class_names.get(class_id, f"Class {class_id}")
        qml_content += (
            f'      <category value="{class_id}" label="{class_name}" '
            f'render="true" symbol="{class_id}"/>\n'
        )

    qml_content += '    </categories>\n    <symbols>\n'

    for class_id in sorted(class_colors.keys()):
        r, g, b = class_colors[class_id][:3]
        qml_content += f'''      <symbol type="fill" name="{class_id}" alpha="{alpha_normalized}" clip_to_extent="1" force_rhr="0">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" name="name" value=""/>
            <Option name="properties"/>
            <Option type="QString" name="type" value="collection"/>
          </Option>
        </data_defined_properties>
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="{r},{g},{b},{alpha}"/>
          <prop k="joinstyle" v="bevel"/>
          <prop k="offset" v="0,0"/>
          <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="offset_unit" v="MM"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.26"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option name="properties"/>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
'''

    qml_content += '    </symbols>\n    <rotation/>\n    <sizescale/>\n  </renderer-v2>\n</qgis>\n'

    output_qml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_qml, 'w', encoding='utf-8') as f:
        f.write(qml_content)

    logger.debug(f"  QML style   : {output_qml}")
    return output_qml


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single_image(input_tif: Path, job_dir: Path,
                         model: torch.nn.Module,
                         target_resolution: Optional[float],
                         tile_size: int, tile_overlap: int,
                         patch_size: int, patch_batch_size: int,
                         use_cuda_tiling: bool, class_field: str,
                         sieve_threshold: int, smooth_tolerance: float,
                         config=Config) -> dict:
    """Process a single image through the complete pipeline."""
    logger = logging.getLogger(__name__)
    start  = datetime.now()
    stem   = input_tif.stem
    logger.info(f"── Processing : {input_tif.name}")

    # Step 1: Resample
    if target_resolution:
        resampled_tif  = job_dir / f"{stem}_resampled.tif"
        resample_tif(input_tif, resampled_tif, target_resolution)
        inference_input = resampled_tif
    else:
        inference_input = input_tif

    # Step 2: Inference
    logger.info(f"  [2/6] Inference ({'CUDA tiling' if use_cuda_tiling else 'CPU'}) ...")
    if use_cuda_tiling and CUPY_AVAILABLE:
        pred_map, profile = cuda_tiling_inference(
            inference_input, model, tile_size, tile_overlap,
            patch_size, patch_batch_size, config=config,
        )
    else:
        raise NotImplementedError("Only CUDA tiling is supported in this pipeline.")

    # Step 3: Raster → Shapefile
    logger.info("  [3/6] Raster → Shapefile ...")
    raw_shp = job_dir / f"{stem}_raw.shp"
    raster_to_shp(pred_map, profile, raw_shp, class_field, config)

    # Step 4: Filter small pixels
    logger.info("  [4/6] Sieve filter ...")
    sieve_shp = job_dir / f"{stem}_sieve.shp"
    try:
        filter_small_pixels(
            str(raw_shp), str(sieve_shp), class_field,
            {"default": sieve_threshold}, 0.3, 4, False,
        )
        input_smooth = str(sieve_shp)
        logger.info(f"    Sieve OK → {sieve_shp.name}")
    except Exception as exc:
        logger.warning(f"    Sieve failed ({exc}), using raw shapefile.")
        input_smooth = str(raw_shp)

    # Step 5: Smooth polygons
    logger.info("  [5/6] Polygon smoothing ...")
    final_shp = job_dir / f"{stem}.shp"
    try:
        smooth_polygons_topojson(input_smooth, str(final_shp), smooth_tolerance)
        logger.info(f"    Smooth OK → {final_shp.name}")
    except Exception as exc:
        logger.warning(f"    Smooth failed ({exc}), copying source directly.")
        for ext in [".shp", ".dbf", ".prj", ".shx", ".cpg"]:
            src = Path(input_smooth).with_suffix(ext)
            if src.exists():
                shutil.copy2(src, final_shp.with_suffix(ext))

    # Step 6: QGIS style
    logger.info("  [6/6] Generating QGIS style ...")
    qml_path = final_shp.with_suffix('.qml')
    create_qml_style(
        config.CLASS_COLORS, config.CLASS_NAMES,
        class_field, qml_path, alpha=200,
    )

    # Cleanup temporaries
    for tmp in [raw_shp, sieve_shp]:
        for ext in [".shp", ".dbf", ".prj", ".shx", ".cpg", ".qpj"]:
            f = tmp.with_suffix(ext)
            if f.exists():
                f.unlink()

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"  ✓ Done in {elapsed:.1f}s → {final_shp.name}")

    return {
        'input'      : str(input_tif),
        'output_shp' : str(final_shp),
        'output_qml' : str(qml_path),
        'time'       : f"{elapsed:.1f}s",
    }


def run_pipeline(input_path: str, model_path: Optional[str] = None,
                 resolution: Optional[float] = None,
                 tile_size: int = 2048, tile_overlap: int = 64,
                 patch_size: int = 512, patch_batch_size: int = 4,
                 use_cuda_tiling: bool = True, class_field: str = "kelas",
                 sieve_threshold: int = 2, smooth_tolerance: float = 0.3,
                 config=Config) -> dict:
    """Bootstrap context, load model, and run the complete pipeline."""

    # --- Bootstrap pipeline context ---
    ctx    = PipelineContext(config)
    logger = ctx.logger

    try:
        input_path = Path(input_path)

        # Discover TIF files
        if input_path.is_file():
            tif_files = [input_path]
        else:
            tif_files = sorted(
                list(input_path.glob('*.tif')) +
                list(input_path.glob('*.tiff'))
            )
        if not tif_files:
            raise ValueError(f"No TIF files found: {input_path}")

        # Create job directory
        job_idx = 1
        while (config.OUTPUT_DIR / f"job_{job_idx}").exists():
            job_idx += 1
        job_dir = config.OUTPUT_DIR / f"job_{job_idx}"
        job_dir.mkdir(parents=True)

        # Resolve model path
        resolved_model = (
            Path(model_path) if model_path
            else config.MODEL_DIR / "best_model.pth"
        )

        log_job_summary(tif_files, job_dir, resolved_model, resolution, config)

        # Load model
        logger.info(f"Loading model from {resolved_model} ...")
        model = SegFormer7Band(config).to(ctx.device)
        ckpt  = torch.load(resolved_model, map_location=ctx.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        log_model_info(model)

        # Process files
        results = []
        for tif_file in tqdm(tif_files, desc="Pipeline", unit="file"):
            try:
                result = process_single_image(
                    tif_file, job_dir, model, resolution,
                    tile_size, tile_overlap, patch_size, patch_batch_size,
                    use_cuda_tiling, class_field,
                    sieve_threshold, smooth_tolerance, config,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed: {tif_file.name} — {exc}", exc_info=True)
                results.append({'input': str(tif_file), 'error': str(exc)})

        total_time = sum(
            float(r['time'].rstrip('s')) for r in results if 'time' in r
        )
        summary = {
            'job_dir' : str(job_dir),
            'total'   : len(tif_files),
            'success' : len([r for r in results if 'error' not in r]),
            'time'    : f"{total_time:.1f}s",
            'results' : results,
        }

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (KeyboardInterrupt).")
        raise

    except Exception as exc:
        logger.exception(f"Fatal pipeline error: {exc}")
        raise

    finally:
        ctx.finalize(summary if 'summary' in dir() else {})

    return summary


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete Pipeline: Resample → Inference → QGIS Style")
    parser.add_argument("--input",        "-i", required=True,       help="TIF file or folder")
    parser.add_argument("--model",        "-m", default=None,        help="Model path (.pth)")
    parser.add_argument("--resolution",   "-r", type=float, default=None, help="Target resolution (m)")
    parser.add_argument("--tile-size",         type=int,   default=2048, help="Tile size (px)")
    parser.add_argument("--tile-overlap",      type=int,   default=64,   help="Tile overlap (px)")
    parser.add_argument("--patch-size",        type=int,   default=512,  help="Patch size (px)")
    parser.add_argument("--patch-batch",       type=int,   default=4,    help="Patch batch size")
    parser.add_argument("--class-field",       type=str,   default="kelas", help="Class field name")
    parser.add_argument("--sieve",             type=int,   default=2,    help="Sieve threshold (px²)")
    parser.add_argument("--smooth",            type=float, default=0.3,  help="Smooth tolerance")
    parser.add_argument("--no-cuda-tiling",    action="store_true",      help="Disable CUDA tiling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = run_pipeline(
        input_path      = args.input,
        model_path      = args.model,
        resolution      = args.resolution,
        tile_size       = args.tile_size,
        tile_overlap    = args.tile_overlap,
        patch_size      = args.patch_size,
        patch_batch_size= args.patch_batch,
        use_cuda_tiling = not args.no_cuda_tiling,
        class_field     = args.class_field,
        sieve_threshold = args.sieve,
        smooth_tolerance= args.smooth,
        config          = Config,
    )
