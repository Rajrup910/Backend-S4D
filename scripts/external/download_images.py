"""Download and unpack ISIC 2019 Training Input (9.1 GB) with progress tracking."""
import urllib.request
import zipfile
import sys
import time
from pathlib import Path

url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
out_dir = Path("data/external")
out_dir.mkdir(parents=True, exist_ok=True)
zip_dest = out_dir / "ISIC_2019_Training_Input.zip"
extract_dir = out_dir / "isic2019_images"

# If already extracted with files, skip
if extract_dir.is_dir() and len(list(extract_dir.glob("*.jpg"))) > 25000:
    print(f"ISIC 2019 images already extracted in {extract_dir} ({len(list(extract_dir.glob('*.jpg')))} files).")
    sys.exit(0)

if not zip_dest.is_file() or zip_dest.stat().st_size < 9_000_000_000:
    print(f"Starting download of ISIC 2019 Training Input from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    t0 = time.time()
    downloaded = 0
    block_size = 1024 * 1024 * 8  # 8 MB blocks
    last_print = 0
    
    with urllib.request.urlopen(req, timeout=120) as resp, open(zip_dest, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        total_gb = total_size / (1024**3)
        print(f"Total size: {total_gb:.2f} GB")
        
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if downloaded - last_print > 500 * 1024 * 1024:  # Print every 500 MB
                elapsed = now - t0
                speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"Downloaded: {downloaded / (1024**3):.2f} / {total_gb:.2f} GB ({downloaded/total_size:.1%}) @ {speed_mb:.1f} MB/s", flush=True)
                last_print = downloaded
                
    elapsed = time.time() - t0
    print(f"Download complete in {elapsed/60:.1f} minutes. File size: {zip_dest.stat().st_size / (1024**3):.2f} GB", flush=True)

print(f"Unpacking {zip_dest} into {extract_dir}...", flush=True)
extract_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_dest, 'r') as zf:
    zf.extractall(extract_dir)

n_extracted = len(list(extract_dir.rglob("*.jpg")))
print(f"Extraction complete. {n_extracted} JPEG images available.", flush=True)
