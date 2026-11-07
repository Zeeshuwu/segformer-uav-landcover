"""
validate_data.py
================
Validasi dan perbaiki data mask yang corrupt.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import Config

def validate_masks(config=Config):
    """
    Cek semua mask files untuk nilai yang invalid.
    """
    masks_dir = config.MASKS_DIR
    all_mask_files = sorted(masks_dir.rglob("*.npy"))
    
    print(f"\n{'='*60}")
    print(f"🔍 VALIDATING {len(all_mask_files)} MASK FILES")
    print(f"{'='*60}\n")
    
    invalid_files = []
    stats = {
        'min_values': [],
        'max_values': [],
        'unique_values': set(),
    }
    
    for mask_path in tqdm(all_mask_files, desc="Validating"):
        try:
            mask = np.load(mask_path)
            
            min_val = mask.min()
            max_val = mask.max()
            unique_vals = np.unique(mask)
            
            stats['min_values'].append(min_val)
            stats['max_values'].append(max_val)
            stats['unique_values'].update(unique_vals.tolist())
            
            # Check for invalid values
            if min_val < 0 or max_val >= config.NUM_CLASSES:
                invalid_files.append({
                    'path': mask_path,
                    'min': min_val,
                    'max': max_val,
                    'unique': unique_vals.tolist(),
                    'shape': mask.shape,
                })
        
        except Exception as e:
            print(f"\n❌ Error loading {mask_path}: {e}")
            invalid_files.append({
                'path': mask_path,
                'error': str(e),
            })
    
    # Print statistics
    print(f"\n{'='*60}")
    print("📊 STATISTICS")
    print(f"{'='*60}")
    print(f"Total files: {len(all_mask_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    print(f"Global min value: {min(stats['min_values'])}")
    print(f"Global max value: {max(stats['max_values'])}")
    print(f"Unique values found: {sorted(stats['unique_values'])}")
    print(f"Expected range: [0, {config.NUM_CLASSES - 1}]")
    
    # Print invalid files
    if invalid_files:
        print(f"\n{'='*60}")
        print("❌ INVALID FILES DETECTED")
        print(f"{'='*60}\n")
        
        for i, info in enumerate(invalid_files[:10], 1):  # Show first 10
            print(f"{i}. {info['path'].name}")
            if 'error' in info:
                print(f"   Error: {info['error']}")
            else:
                print(f"   Min: {info['min']}, Max: {info['max']}")
                print(f"   Unique: {info['unique']}")
                print(f"   Shape: {info['shape']}")
            print()
        
        if len(invalid_files) > 10:
            print(f"... and {len(invalid_files) - 10} more files\n")
    
    return invalid_files, stats


def fix_masks(invalid_files, config=Config, backup=True):
    """
    Perbaiki mask files dengan clipping nilai ke range valid.
    """
    if not invalid_files:
        print("✅ No files to fix!")
        return
    
    print(f"\n{'='*60}")
    print(f"🔧 FIXING {len(invalid_files)} FILES")
    print(f"{'='*60}\n")
    
    ans = input("Proceed with fixing? This will modify files! [y/N]: ").strip().lower()
    if ans != 'y':
        print("Aborted.")
        return
    
    backup_dir = config.MASKS_DIR.parent / "masks_backup"
    if backup:
        backup_dir.mkdir(exist_ok=True)
        print(f"📦 Backup directory: {backup_dir}\n")
    
    for info in tqdm(invalid_files, desc="Fixing"):
        if 'error' in info:
            continue
        
        mask_path = info['path']
        
        try:
            # Load
            mask = np.load(mask_path)
            
            # Backup
            if backup:
                backup_path = backup_dir / mask_path.name
                np.save(backup_path, mask)
            
            # Fix: Clip to valid range
            mask_fixed = np.clip(mask, 0, config.NUM_CLASSES - 1)
            
            # Save
            np.save(mask_path, mask_fixed.astype(np.int64))
            
        except Exception as e:
            print(f"\n❌ Error fixing {mask_path}: {e}")
    
    print("\n✅ Fixing complete!")


def check_ignore_index(config=Config):
    """
    Cek apakah ada nilai IGNORE_INDEX (255) di mask.
    """
    masks_dir = config.MASKS_DIR
    all_mask_files = sorted(masks_dir.rglob("*.npy"))
    
    print(f"\n{'='*60}")
    print(f"🔍 CHECKING FOR IGNORE_INDEX ({config.IGNORE_INDEX})")
    print(f"{'='*60}\n")
    
    files_with_ignore = []
    
    for mask_path in tqdm(all_mask_files, desc="Checking"):
        mask = np.load(mask_path)
        if config.IGNORE_INDEX in mask:
            count = (mask == config.IGNORE_INDEX).sum()
            files_with_ignore.append({
                'path': mask_path,
                'count': count,
                'percentage': count / mask.size * 100,
            })
    
    if files_with_ignore:
        print(f"\n⚠️  Found {len(files_with_ignore)} files with IGNORE_INDEX\n")
        for info in files_with_ignore[:5]:
            print(f"  {info['path'].name}: {info['count']} pixels ({info['percentage']:.2f}%)")
    else:
        print(f"\n✅ No IGNORE_INDEX found")
    
    return files_with_ignore


if __name__ == "__main__":
    config = Config
    
    # Step 1: Validate
    invalid_files, stats = validate_masks(config)
    
    # Step 2: Check ignore index
    ignore_files = check_ignore_index(config)
    
    # Step 3: Fix if needed
    if invalid_files:
        fix_masks(invalid_files, config, backup=True)
        
        # Re-validate
        print("\n" + "="*60)
        print("🔄 RE-VALIDATING AFTER FIX")
        print("="*60)
        invalid_files, stats = validate_masks(config)
