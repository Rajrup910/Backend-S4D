"""Download ISIC 2019 metadata and ground truth tables."""
import urllib.request
from pathlib import Path

out_dir = Path("data/external")
out_dir.mkdir(parents=True, exist_ok=True)

files = {
    "ISIC_2019_Training_Metadata.csv": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
    "ISIC_2019_Training_GroundTruth.csv": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
}

for fname, url in files.items():
    dest = out_dir / fname
    if not dest.is_file() or dest.stat().st_size == 0:
        print(f"Downloading {fname} from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        print(f"Saved {fname} ({dest.stat().st_size} bytes)")
    else:
        print(f"{fname} already exists ({dest.stat().st_size} bytes)")
