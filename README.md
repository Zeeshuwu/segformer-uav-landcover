# segformer-uav-landcover

Land cover segmentation of UAV RGB orthomosaics with SegFormer. The model takes three RGB bands plus four channels computed from them: two vegetation indices, an edge map and a texture map.

## Features

- **Seven-channel input built from RGB:**
  - R, G, B.
  - Excess Green index (ExG) and Visible Atmospherically Resistant Index (VARI).
  - Canny edge map.
  - Local Binary Pattern (LBP) texture map.
- **SegFormer-B3 with a channel-adaptive stem.** The pretrained patch-embedding layer is extended to any number of input channels while keeping its pretrained RGB weights. `--bands` chooses which channels to train on.
- **Spatial-block data split.** Orthomosaics are divided into 2,048 px blocks before patching, and each block goes entirely to train, validation or test. Patches from neighbouring areas therefore never end up in different splits.
- **Imbalance-aware training:**
  - Combined focal and Dice loss with class weights.
  - Class balancing on the training split only.
  - Cosine learning-rate schedule and early stopping.
- **Large-image inference:**
  - Tiled inference with Gaussian-weighted blending of overlapping patches.
  - Resampling to the training resolution.
  - Export to raster and polygons, with small-region filtering and topology-preserving smoothing.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings: paths, bands, classes, patch size, model, training and inference parameters. |
| `build_training_dataset.py` | Rasterises label polygons, builds the seven-channel patches and makes the spatial-block split. |
| `train.py` | Trains SegFormer on the chosen bands. |
| `inference.py` | Runs a trained model on a GeoTIFF or a folder of GeoTIFFs and exports raster and polygon results. |
| `filter_small_pixels.py` | Removes small regions from a polygon layer (sieve). It can also be used on its own. |
| `smoothing_topojson.py` | Smooths polygon boundaries without opening gaps between neighbours. |
| `validate_data.py` | Checks and repairs corrupt mask patches. |

## Usage

```bash
pip install -r requirements.txt

# 1. Training data: orthomosaics (GeoTIFF) and label polygons (shapefile, class in field "ID")
python build_training_dataset.py --tif_dir data/raw --shp_dir data/raw

# 2. Training (all seven channels, or e.g. --bands R,G,B for an RGB baseline)
python train.py --bands R,G,B,ExG,VARI,Edge,Texture --seed 42

# 3. Inference on new imagery
python inference.py --input image.tif --model path/to/best_model.pth --resolution 0.1
```

Data and outputs are stored outside the repository. Set the `SEGFORMER_WORKDIR` environment variable to the working folder; by default the repository folder is used.

## Classes

The default legend has six land cover classes: 1 built-up (Permukiman), 2 water (Badan Air), 3 high vegetation (Vegetasi Tinggi), 4 low vegetation (Vegetasi Rendah), 5 road (Jalan) and 6 bare land (Tanah Terbuka). Class 0 is background and is not used; unlabelled pixels are 255 and ignored. Change `CLASS_NAMES` and `NUM_CLASSES` in `config.py` for a different legend.

## Background

This code was prepared for a study presented at FIT ISI × Geodeta 2026 and submitted to *IOP Conference Series: Earth and Environmental Science*. The repository contains the general method, not the study's imagery, experiments or results. The one exception is the study's patch-to-split list, `study_split/split_list.csv`, so that readers can check the spatial split.

The list has one row per 512 × 512 px patch (2,882 rows). Its columns are:
- `patch_id`: patch identifier.
- `orthomosaic`: A or B.
- `block_id`: the 2,048 px block that contains the patch.
- `split`: training, validation or test.
- `row_px`, `col_px`: the patch's upper-left pixel in its 0.10 m orthomosaic (EPSG:32650). The upper-left corner of orthomosaic A is at E 551,834.43 m, N 21,251.33 m; that of B is at E 553,380.53 m, N 12,998.93 m.
- `labelled_px`: labelled pixels in the patch.
- `dominant_class`: the most frequent class in the patch.
- `used`: whether the patch was used, or removed from the training split by class balancing.

Every block belongs to exactly one split.

## License

The code is released under the MIT License (see `LICENSE`). The pretrained SegFormer weights (`nvidia/segformer-b3-finetuned-ade-512-512`) are downloaded from Hugging Face and are covered by NVIDIA's own license, which restricts commercial use.
