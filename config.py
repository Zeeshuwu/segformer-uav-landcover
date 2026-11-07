import os
from pathlib import Path
from rasterio.warp import Resampling


class Config:
    BASE_DIR            = Path(__file__).parent
    # Data and outputs live outside the repo; set SEGFORMER_WORKDIR to point the
    # whole pipeline at another working folder.
    WORK_DIR            = Path(os.environ.get("SEGFORMER_WORKDIR", BASE_DIR))
    DATA_DIR            = WORK_DIR / "data"
    RAW_DATA_DIR        = DATA_DIR / "raw"
    PROCESSED_DATA_DIR  = DATA_DIR / "processed"
    OUTPUT_DIR          = WORK_DIR / "outputs"

    CLAHE_OUTPUT_DIR    = PROCESSED_DATA_DIR / "clahe_enhanced"
    PATCHES_DIR         = PROCESSED_DATA_DIR / "patches"
    MASKS_DIR           = PROCESSED_DATA_DIR / "masks"
    LABEL_RASTER_DIR    = PROCESSED_DATA_DIR / "label_rasters"
    SPLIT_MANIFEST      = PROCESSED_DATA_DIR / "split_manifest.json"
    RUNS_DIR            = OUTPUT_DIR / "runs"

    MODEL_DIR           = OUTPUT_DIR / "models"
    LOG_DIR             = OUTPUT_DIR / "logs"
    CHECKPOINT_DIR      = OUTPUT_DIR / "checkpoints"
    RESULTS_DIR         = OUTPUT_DIR / "results"

    CLAHE_REF_SITE      = None
    KELAS_COL           = "ID"

    TARGET_EPSG         = 32650
    TARGET_GSD          = 0.1
    RESAMPLING_METHOD   = Resampling.bilinear

    CLAHE_TILE_GRID     = (8, 8)
    CLAHE_CLIP_DEFAULT  = 2.0
    CLAHE_CLIP_BOUNDS   = (1.0, 15.0)
    CLAHE_CLIP_XTOL     = 0.2
    CLAHE_BLEND         = True
    CLAHE_SKIP_EXISTING = True

    TIFF_REPROJ_REF     = PROCESSED_DATA_DIR / "_ref_reproj.tif"

    CONFUSING_PAIRS = [
        (1, 2),
        (2, 6),
        (4, 5),
    ]

    IMAGE_EXT       = ".tif"
    SHAPEFILE_EXT   = ".shp"

    BAND_NAMES      = ['R', 'G', 'B', 'ExG', 'VARI', 'Edge', 'Texture']
    NUM_CHANNELS    = 7

    NUM_CLASSES     = 7
    # Index 0 (background) never occurs in the labels (unlabelled pixels are 255),
    # so every metric is a macro mean over the six land-cover classes only.
    EVAL_CLASSES    = [1, 2, 3, 4, 5, 6]
    
    CLASS_COLORS = {
        0: [0, 0, 0],
        1: [139, 0, 0],
        2: [72, 209, 204],
        3: [34, 139, 34],
        4: [189, 252, 201],
        5: [255, 215, 0],
        6: [244, 164, 96],
    }
    
    CLASS_NAMES = {
        0: 'Background',
        1: 'Permukiman',
        2: 'Badan Air',
        3: 'Vegetasi Tinggi',
        4: 'Vegetasi Rendah',
        5: 'Jalan',
        6: 'Tanah Terbuka',
    }

    PATCH_SIZE          = 512
    OVERLAP             = 128
    STRIDE              = PATCH_SIZE - OVERLAP

    SAVE_DTYPE_IMAGE    = "float16"
    SAVE_DTYPE_MASK     = "uint8"

    NORMALIZE           = True
    NORMALIZE_MEAN      = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5]
    NORMALIZE_STD       = [0.229, 0.224, 0.225, 0.25, 0.25, 0.25, 0.25]

    MIN_VALID_PIXELS    = 0.3
    IGNORE_INDEX        = 255

    MODEL_NAME          = "nvidia/segformer-b3-finetuned-ade-512-512"
    ENCODER_NAME        = "mit_b3"
    PRETRAINED          = True
    FREEZE_ENCODER      = False
    DECODER_HIDDEN_SIZE = 256

    BATCH_SIZE          = 8
    NUM_EPOCHS          = 100
    LEARNING_RATE       = 2e-5
    WEIGHT_DECAY        = 0.05

    OPTIMIZER           = "adamw"
    BETAS               = (0.9, 0.999)
    EPS                 = 1e-8

    LR_SCHEDULER        = "cosine"
    LR_POWER            = 0.9
    WARMUP_EPOCHS       = 10
    WARMUP_LR           = 1e-6

    LOSS_FUNCTION       = "focal_dice"
    
    FOCAL_ALPHA         = 0.25
    FOCAL_GAMMA         = 3.0
    
    DICE_SMOOTH         = 1.0
    
    FOCAL_WEIGHT        = 0.3
    DICE_WEIGHT         = 0.7
    
    # None -> read the weights computed on the training split from SPLIT_MANIFEST.
    CLASS_WEIGHTS = None

    USE_AUGMENTATION        = True
    AUG_HORIZONTAL_FLIP     = 0.5
    AUG_VERTICAL_FLIP       = 0.5
    AUG_ROTATION            = 0.5
    AUG_ROTATION_LIMIT      = 90
    AUG_BRIGHTNESS          = 0.3
    AUG_CONTRAST            = 0.3

    # ============ SPLIT & CROSS VALIDATION ============
    TRAIN_SPLIT             = 0.7
    VAL_SPLIT               = 0.3
    RANDOM_SEED             = 42

    # Spatial split: each orthomosaic is cut into square blocks before patching and
    # whole blocks go to train / val / test, so no pixel is shared between splits.
    # Shares are of labelled pixels. The split seed stays fixed across model seeds.
    SPLIT_BLOCK_SIZE        = 2048
    SPLIT_FRACTIONS         = {"train": 0.6, "val": 0.2, "test": 0.2}
    SPLIT_SEED              = 42
    SPLIT_SEARCH_ITERS      = 5000
    
    # K-Fold Cross Validation (untuk dataset kecil < 2000 samples)
    USE_KFOLD               = False
    KFOLD_SPLITS            = 5
    
    # ============ EARLY STOPPING ============
    EARLY_STOPPING          = True
    EARLY_STOPPING_PATIENCE = 10    # Stop jika tidak improve selama 15 epoch
    EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement
    
    VAL_FREQUENCY           = 1
    SAVE_BEST_ONLY          = False

    LOG_FREQUENCY   = 10
    TENSORBOARD     = False
    WANDB           = False
    WANDB_PROJECT   = "segformer-7band-landcover"
    SAVE_FREQUENCY  = 5
    MAX_CHECKPOINTS = 5

    TEST_TIME_AUGMENTATION  = False
    INFERENCE_BATCH_SIZE    = 8
    OVERLAP_INFERENCE       = 64

    DEVICE          = "cuda"
    NUM_WORKERS     = 4
    PIN_MEMORY      = True
    MIXED_PRECISION = True
    MULTI_GPU       = False
    GPU_IDS         = [0]

    # 1 = optimizer step every batch (effective batch 8, as in the original runs).
    GRADIENT_ACCUMULATION_STEPS = 1

    @classmethod
    def make_dirs(cls):
        for d in [
            cls.PROCESSED_DATA_DIR,
            cls.CLAHE_OUTPUT_DIR,
            cls.PATCHES_DIR,
            cls.MASKS_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.CHECKPOINT_DIR,
            cls.RESULTS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


    @classmethod
    def diagnose(cls):
        """Cetak info struktur folder & file yang ditemukan."""
        print("\n" + "="*60)
        print("🔍 CONFIG DIAGNOSIS")
        print("="*60)
        print(f"  BASE_DIR       : {cls.BASE_DIR}")
        print(f"  RAW_DATA_DIR   : {cls.RAW_DATA_DIR}")
        print(f"  CLAHE_REF_SITE : {cls.CLAHE_REF_SITE}")
        print(f"  KELAS_COL      : {cls.KELAS_COL}")
        print()

        if not cls.RAW_DATA_DIR.exists():
            print(f"  ❌ RAW_DATA_DIR tidak ditemukan: {cls.RAW_DATA_DIR}")
            print(f"     → Buat folder: {cls.RAW_DATA_DIR}")
            return

        sites = sorted([d for d in cls.RAW_DATA_DIR.iterdir() if d.is_dir()])
        if not sites:
            print(f"  ❌ Tidak ada sub-folder site di {cls.RAW_DATA_DIR}")
            print(f"     → Struktur yang diharapkan:")
            print(f"        data/raw/")
            print(f"        └── nama_site/")
            print(f"            ├── gambar.tif")
            print(f"            └── label.shp")
            return

        print(f"  ✅ Ditemukan {len(sites)} site:")
        for site in sites:
            tifs = list(site.glob("*.tif"))
            shps = list(site.glob("*.shp"))
            tif_ok = "✅" if tifs else "❌"
            shp_ok = "✅" if shps else "❌"
            print(f"     📁 {site.name}")
            print(f"        {tif_ok} TIF : {len(tifs)} file(s)")
            print(f"        {shp_ok} SHP : {len(shps)} file(s)")

            if shps:
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(shps[0])
                    cols = list(gdf.columns)
                    kelas_ok = "✅" if cls.KELAS_COL in cols else "❌"
                    print(f"        {kelas_ok} Kolom '{cls.KELAS_COL}' "
                          f"{'ditemukan' if cls.KELAS_COL in cols else 'TIDAK ADA'}")
                    if cls.KELAS_COL not in cols:
                        print(f"           Kolom tersedia: {cols}")
                except Exception as e:
                    print(f"        ⚠️  Gagal baca SHP: {str(e)[:50]}")

        print()
        if cls.CLAHE_REF_SITE is None:
            print(f"  💡 CLAHE_REF_SITE = None → akan pakai '{sites[0].name}' sebagai referensi")
        else:
            ref_dir = cls.RAW_DATA_DIR / cls.CLAHE_REF_SITE
            if ref_dir.exists():
                print(f"  ✅ CLAHE_REF_SITE '{cls.CLAHE_REF_SITE}' ditemukan")
            else:
                print(f"  ❌ CLAHE_REF_SITE '{cls.CLAHE_REF_SITE}' TIDAK ditemukan!")
                print(f"     → Set CLAHE_REF_SITE = None untuk otomatis,")
                print(f"       atau ganti ke salah satu: {[s.name for s in sites]}")
        print("="*60 + "\n")

    @classmethod
    def summary(cls):
        """Print ringkasan konfigurasi training."""
        print("\n" + "="*60)
        print("⚙️  TRAINING CONFIGURATION SUMMARY")
        print("="*60)
        print(f"  Model Architecture : {cls.MODEL_NAME}")
        print(f"  Input Channels     : {cls.NUM_CHANNELS} ({', '.join(cls.BAND_NAMES)})")
        print(f"  Number of Classes  : {cls.NUM_CLASSES}")
        print(f"  Patch Size         : {cls.PATCH_SIZE}x{cls.PATCH_SIZE}")
        print(f"  Overlap            : {cls.OVERLAP}")
        print()
        print("  Training Parameters:")
        print(f"    Batch Size       : {cls.BATCH_SIZE}")
        print(f"    Epochs           : {cls.NUM_EPOCHS}")
        print(f"    Learning Rate    : {cls.LEARNING_RATE}")
        print(f"    Optimizer        : {cls.OPTIMIZER.upper()}")
        print(f"    Weight Decay     : {cls.WEIGHT_DECAY}")
        print(f"    LR Scheduler     : {cls.LR_SCHEDULER}")
        print(f"    Warmup Epochs    : {cls.WARMUP_EPOCHS}")
        print()
        print("  Loss Function:")
        print(f"    Type             : {cls.LOSS_FUNCTION.upper()}")
        if cls.LOSS_FUNCTION == "focal_dice":
            print(f"    Focal Weight     : {cls.FOCAL_WEIGHT}")
            print(f"    Dice Weight      : {cls.DICE_WEIGHT}")
            print(f"    Focal Alpha      : {cls.FOCAL_ALPHA}")
            print(f"    Focal Gamma      : {cls.FOCAL_GAMMA}")
            print(f"    Dice Smooth      : {cls.DICE_SMOOTH}")
        print()
        print("  Hardware:")
        print(f"    Device           : {cls.DEVICE}")
        print(f"    Mixed Precision  : {cls.MIXED_PRECISION}")
        print(f"    Num Workers      : {cls.NUM_WORKERS}")
        print(f"    Pin Memory       : {cls.PIN_MEMORY}")
        print()
        print("  Augmentation:")
        print(f"    Enabled          : {cls.USE_AUGMENTATION}")
        if cls.USE_AUGMENTATION:
            print(f"    H-Flip (p)       : {cls.AUG_HORIZONTAL_FLIP}")
            print(f"    V-Flip (p)       : {cls.AUG_VERTICAL_FLIP}")
            print(f"    Rotation (p)     : {cls.AUG_ROTATION} (±{cls.AUG_ROTATION_LIMIT}°)")
            print(f"    Brightness       : {cls.AUG_BRIGHTNESS}")
            print(f"    Contrast         : {cls.AUG_CONTRAST}")
        print()
        print("  Validation & Early Stopping:")
        print(f"    Val Split        : {cls.VAL_SPLIT}")
        print(f"    Use K-Fold       : {cls.USE_KFOLD}")
        if cls.USE_KFOLD:
            print(f"    K-Fold Splits    : {cls.KFOLD_SPLITS}")
        print(f"    Early Stopping   : {cls.EARLY_STOPPING}")
        if cls.EARLY_STOPPING:
            print(f"    ES Patience      : {cls.EARLY_STOPPING_PATIENCE} epochs")
            print(f"    ES Min Delta     : {cls.EARLY_STOPPING_MIN_DELTA}")
        print()
        print("  Checkpointing:")
        print(f"    Save Frequency   : Every {cls.SAVE_FREQUENCY} epoch(s)")
        print(f"    Max Checkpoints  : {cls.MAX_CHECKPOINTS}")
        print(f"    TensorBoard      : {cls.TENSORBOARD}")
        print("="*60 + "\n")

    @classmethod
    def validate_config(cls):
        """Validasi konfigurasi sebelum training."""
        errors = []
        warnings = []

        # Validasi loss weights
        if cls.LOSS_FUNCTION == "focal_dice":
            total_weight = cls.FOCAL_WEIGHT + cls.DICE_WEIGHT
            if abs(total_weight - 1.0) > 0.01:
                warnings.append(
                    f"FOCAL_WEIGHT + DICE_WEIGHT = {total_weight:.2f} (disarankan = 1.0)"
                )

        # Validasi batch size
        if cls.BATCH_SIZE < 1:
            errors.append("BATCH_SIZE harus >= 1")

        # Validasi num_classes
        if cls.NUM_CLASSES != len(cls.CLASS_COLORS):
            errors.append(
                f"NUM_CLASSES ({cls.NUM_CLASSES}) != len(CLASS_COLORS) ({len(cls.CLASS_COLORS)})"
            )

        # Validasi paths
        if not cls.RAW_DATA_DIR.exists():
            warnings.append(f"RAW_DATA_DIR tidak ditemukan: {cls.RAW_DATA_DIR}")

        # Validasi K-Fold
        if cls.USE_KFOLD and cls.KFOLD_SPLITS < 2:
            errors.append("KFOLD_SPLITS harus >= 2")

        # Validasi Early Stopping
        if cls.EARLY_STOPPING and cls.EARLY_STOPPING_PATIENCE < 1:
            errors.append("EARLY_STOPPING_PATIENCE harus >= 1")

        # Print hasil validasi
        if errors:
            print("\n" + "="*60)
            print("❌ CONFIGURATION ERRORS:")
            for err in errors:
                print(f"  • {err}")
            print("="*60 + "\n")
            raise ValueError("Konfigurasi tidak valid! Perbaiki error di atas.")

        if warnings:
            print("\n" + "="*60)
            print("⚠️  CONFIGURATION WARNINGS:")
            for warn in warnings:
                print(f"  • {warn}")
            print("="*60 + "\n")

        return True
