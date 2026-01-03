# Data Download Instructions

This project uses the NIH Chest X-Rayp8 dataset. The `download_data.py` script handles automated downloads.

## Automated Setup (CSV Only)

✅ The CSV annotation file (`Data_Entry_2017.csv`) has been downloaded automatically.

## Manual Image Download (Required)

The image files (~4GB) need to be downloaded manually:

1. **Visit**: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
   
2. **Download**: `images_002.tar.gz` (~4GB)
   
3. **Extract** to: `data/images_002/`
   
4. **Verify** structure:
   ```
   data/
   ├── Data_Entry_2017.csv  ✅ Already downloaded
   └── images_002/
       └── images/
           ├── 00000001_000.png
           ├── 00000001_001.png
           └── ... (~10,000 PNG files)
   ```

## Alternative: Full Kaggle Download

If you prefer to download from Kaggle (not recommended due to 45GB size):
```bash
kaggle datasets download -d nih-chest-xrays/data --unzip
```

## Verification

Run the download script to check your setup:
```bash
python download_data.py
```

It will verify both CSV and images are in the correct location.
