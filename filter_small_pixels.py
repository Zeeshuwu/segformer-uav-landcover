"""
filter_small_pixels.py
======================
Rasterize → Sieve small connected components → Polygonize kembali ke SHP.

Dipakai oleh inference.py sebagai post-processing step setelah polygonize.
Bisa juga dijalankan standalone untuk SHP apapun.

Usage (standalone):
    python filter_small_pixels.py --input path/to/input.shp
    python filter_small_pixels.py --input path/to/input.shp \\
        --output path/to/output.shp \\
        --class-field kelas \\
        --default-threshold 100 \\
        --pixel-size 0.5 \\
        --connectivity 4
"""

import argparse
import math
import os
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import Affine
from rasterio.features import rasterize, shapes, sieve
from rasterio.transform import from_origin
from shapely.geometry import shape as shapely_shape


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _compute_grid_from_bounds(
    bounds: Tuple[float, float, float, float],
    pixel_size: float,
) -> Tuple[Affine, int, int]:
    """Hitung parameter raster grid dari bounds vektor."""
    minx, miny, maxx, maxy = bounds
    width     = int(math.ceil((maxx - minx) / pixel_size))
    height    = int(math.ceil((maxy - miny) / pixel_size))
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    return transform, width, height


def _determine_dtype_and_nodata(values: np.ndarray) -> Tuple[np.dtype, int]:
    """Tentukan dtype dan nodata yang tepat untuk rasterisasi."""
    max_val = int(values.max()) if len(values) > 0 else 1
    min_val = int(values.min()) if len(values) > 0 else 1

    if min_val <= 0:
        # Ada kelas bernilai 0 → pakai signed dtype dengan nodata = -1
        nodata = -1
        dtype  = np.int16 if max_val <= np.iinfo(np.int16).max else np.int32
    else:
        nodata = 0
        if max_val <= np.iinfo(np.uint8).max:
            dtype = np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif max_val <= np.iinfo(np.int16).max:
            dtype = np.int16
        else:
            dtype = np.int32

    return dtype, nodata


def _prepare_class_values(
    gdf: gpd.GeoDataFrame,
    class_field: str,
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Ekstrak nilai kelas sebagai integer.
    String → factorize ke integer, simpan lookup untuk restore.
    """
    label_lookup = None
    series       = gdf[class_field]

    if series.dtype.kind in ("i", "u"):
        values = series.astype(int).to_numpy()
    elif series.dtype.kind == "f":
        values = series.round().astype(int).to_numpy()
    else:
        # String → integer codes
        codes, uniques = pd.factorize(series, sort=True)
        values         = codes.astype(int)
        label_lookup   = {
            int(code): str(label)
            for code, label in enumerate(uniques)
        }

    return values, label_lookup


def _ensure_sieve_compatible_dtype(arr: np.ndarray) -> np.ndarray:
    """Cast array ke dtype yang kompatibel dengan rasterio.sieve."""
    if arr.dtype not in (np.uint8, np.uint16, np.int16, np.int32):
        if arr.max() <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16, copy=False)
        return arr.astype(np.int32, copy=False)
    return arr


def _sieve_base_layer(
    raster: np.ndarray,
    threshold: int,
    connectivity: int,
) -> np.ndarray:
    """Sieve seluruh raster dengan threshold default."""
    if threshold <= 1:
        return raster.copy()
    r = _ensure_sieve_compatible_dtype(raster)
    return sieve(r, size=threshold, connectivity=connectivity,
                 out=np.zeros_like(r))


def _sieve_single_class(
    raster: np.ndarray,
    class_id: int,
    threshold: int,
    connectivity: int,
    nodata: int,
) -> Optional[np.ndarray]:
    """
    Sieve satu kelas spesifik dari raster.
    Returns None jika kelas tidak ada di raster.
    """
    class_mask = raster == class_id
    if not np.any(class_mask):
        return None

    class_raster = np.where(class_mask, raster, nodata)
    class_raster = _ensure_sieve_compatible_dtype(class_raster)

    return sieve(
        class_raster,
        size=threshold,
        connectivity=connectivity,
        out=np.zeros_like(class_raster),
    )


def _polygonize(
    raster: np.ndarray,
    transform: Affine,
    nodata: int,
    class_field: str,
    label_lookup: Optional[dict],
) -> list:
    """Polygonize raster → list of record dict."""
    records = []
    for geom_mapping, value in shapes(
        raster,
        mask=(raster != nodata).astype(np.uint8),
        transform=transform,
    ):
        if int(value) == nodata:
            continue
        field_val = (
            label_lookup.get(int(value), int(value))
            if label_lookup is not None
            else int(value)
        )
        records.append({
            "geometry" : shapely_shape(geom_mapping),
            class_field: field_val,
        })
    return records


# ============================================================
# PUBLIC API
# ============================================================

def filter_small_pixels(
    shp_path            : str,
    output_shp_path     : Optional[str]  = None,
    class_field         : str            = "kelas",
    class_thresholds    : Optional[dict] = None,
    inferred_pixel_size : float          = 1.0,
    connectivity        : int            = 4,
    all_touched         : bool           = False,
) -> str:
    """
    Rasterize SHP → sieve komponen kecil → polygonize kembali ke SHP.

    Parameters
    ----------
    shp_path : str
        Input SHP (hasil polygonize dari inference atau SHP apapun).
    output_shp_path : str, optional
        Path output SHP. Jika None, otomatis dibuat di folder
        parent dengan suffix "_sieve".
    class_field : str
        Nama kolom kelas di SHP. Default "kelas" (sesuai inference.py).
    class_thresholds : dict, optional
        Threshold per kelas dalam satuan pixel count.
        Format: {"default": 100, 1: 50, 4: 200}
        - "default" : threshold base layer (wajib ada)
        - int key   : threshold override untuk kelas tertentu
        Kelas dengan threshold lebih kecil = prioritas lebih tinggi
        (ditimpa terakhir → tidak tersieve).
        Default: {"default": 100}
    inferred_pixel_size : float
        Ukuran pixel untuk rasterisasi (meter). Sesuaikan dengan GSD data.
        Default 1.0 meter.
    connectivity : int
        4 = hanya kiri/kanan/atas/bawah.
        8 = termasuk diagonal.
        Default 4.
    all_touched : bool
        True = bakar semua pixel yang disentuh geometri.
        False = hanya pixel yang pusatnya di dalam geometri.
        Default False.

    Returns
    -------
    str
        Path ke output SHP yang disimpan.
    """

    # ── Validasi input ──────────────────────────────────────────
    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(f"Input SHP kosong: {shp_path}")
    if gdf.crs is None:
        raise ValueError(f"Input SHP tidak memiliki CRS: {shp_path}")
    if class_field not in gdf.columns:
        raise ValueError(
            f"Kolom '{class_field}' tidak ditemukan di SHP.\n"
            f"Kolom tersedia: {list(gdf.columns)}\n"
            f"→ Sesuaikan parameter class_field."
        )
    if gdf.crs.is_geographic:
        raise ValueError(
            "CRS geografis terdeteksi. Reproject ke projected CRS (meter) terlebih dahulu."
        )

    print(f"    📊 Input: {len(gdf)} polygon, CRS: {gdf.crs.to_epsg()}")

    # ── Setup grid raster ───────────────────────────────────────
    transform, width, height = _compute_grid_from_bounds(
        gdf.total_bounds, inferred_pixel_size
    )
    print(f"    🗺️  Raster grid: {height}×{width} px "
          f"@ {inferred_pixel_size}m/px")

    # ── Persiapan nilai kelas ───────────────────────────────────
    values, label_lookup = _prepare_class_values(gdf, class_field)

    # Fix geometri invalid sebelum rasterisasi
    geoms          = gdf.geometry.buffer(0)
    shapes_to_burn = [
        (geom, int(val))
        for geom, val in zip(geoms, values)
        if geom is not None and not geom.is_empty
    ]
    if not shapes_to_burn:
        raise ValueError("Tidak ada geometri valid untuk dirasterisasi.")

    dtype, nodata = _determine_dtype_and_nodata(values)

    # ── Rasterisasi ─────────────────────────────────────────────
    raster = rasterize(
        shapes    = shapes_to_burn,
        out_shape = (height, width),
        transform = transform,
        fill      = nodata,
        dtype     = dtype,
        all_touched = all_touched,
    )

    unique_before = np.unique(raster[raster != nodata])
    print(f"    🎨 Kelas di raster: {unique_before.tolist()}")

    # ── Sieve ───────────────────────────────────────────────────
    if class_thresholds is None:
        class_thresholds = {"default": 100}

    default_threshold = class_thresholds.get("default", 100)

    # Step 1: Base layer — sieve semua kelas dengan default threshold
    print(f"    🧹 Base sieve threshold: {default_threshold} px")
    final_raster = _sieve_base_layer(raster, default_threshold, connectivity)

    # Step 2: Per-class override
    # Urutkan dari threshold BESAR ke KECIL:
    # threshold besar diproses duluan → threshold kecil ditimpa terakhir
    # → kelas dengan threshold kecil punya prioritas lebih tinggi (lebih sedikit tersieve)
    override_classes = sorted(
        [
            (class_id, thr)
            for class_id, thr in class_thresholds.items()
            if class_id != "default"
        ],
        key=lambda x: x[1],
        reverse=True,   # besar dulu
    )

    for class_id, thr in override_classes:
        print(f"    🔧 Override kelas {class_id}: threshold={thr} px")
        sieved = _sieve_single_class(raster, class_id, thr, connectivity, nodata)
        if sieved is not None:
            final_raster = np.where(sieved != nodata, sieved, final_raster)

    unique_after = np.unique(final_raster[final_raster != nodata])
    print(f"    ✅ Kelas setelah sieve: {unique_after.tolist()}")

    # ── Polygonize ──────────────────────────────────────────────
    records = _polygonize(final_raster, transform, nodata, class_field, label_lookup)

    if not records:
        raise ValueError(
            "Tidak ada polygon yang dihasilkan setelah sieve.\n"
            "Coba turunkan nilai threshold."
        )

    out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)

    # Dissolve polygon bersebelahan dengan kelas sama → kurangi fragment
    out_gdf = out_gdf.dissolve(by=class_field, as_index=False)
    out_gdf = out_gdf.explode(index_parts=False).reset_index(drop=True)

    print(f"    📦 Output: {len(out_gdf)} polygon")

    # ── Resolve output path ─────────────────────────────────────
    if output_shp_path is None:
        in_dir      = os.path.dirname(os.path.abspath(shp_path))
        in_basename = os.path.basename(in_dir)
        parent_dir  = os.path.dirname(in_dir)
        out_dir     = os.path.join(parent_dir, f"{in_basename}_sieve")
        os.makedirs(out_dir, exist_ok=True)
        in_stem, _  = os.path.splitext(os.path.basename(shp_path))
        output_shp_path = os.path.join(out_dir, f"{in_stem}_sieve.shp")
    else:
        out_dir = os.path.dirname(output_shp_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    out_gdf.to_file(output_shp_path)
    print(f"    💾 Sieve output: {output_shp_path}")

    return output_shp_path


# ============================================================
# CLI ENTRY POINT
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Filter small pixels dari SHP segmentasi via rasterize → sieve → polygonize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path ke SHP input"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path ke SHP output. Default: otomatis dengan suffix _sieve"
    )
    parser.add_argument(
        "--class-field", "-c", default="kelas",
        help="Nama kolom kelas di SHP"
    )
    parser.add_argument(
        "--default-threshold", "-t", type=int, default=100,
        help="Threshold default (pixel count). Komponen < threshold dihapus"
    )
    parser.add_argument(
        "--pixel-size", "-p", type=float, default=1.0,
        help="Ukuran pixel untuk rasterisasi (meter, sesuai GSD data)"
    )
    parser.add_argument(
        "--connectivity", type=int, default=4, choices=[4, 8],
        help="Konektivitas sieve: 4=4-arah, 8=8-arah"
    )
    parser.add_argument(
        "--all-touched", action="store_true",
        help="Bakar semua pixel yang disentuh geometri (bukan hanya pusat)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    filter_small_pixels(
        shp_path            = args.input,
        output_shp_path     = args.output,
        class_field         = args.class_field,
        class_thresholds    = {"default": args.default_threshold},
        inferred_pixel_size = args.pixel_size,
        connectivity        = args.connectivity,
        all_touched         = args.all_touched,
    )