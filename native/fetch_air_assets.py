"""Fetch the author's public AIR/POCO weights and one public observed cloud."""
import argparse
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

import numpy as np

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/sicongpan/air-osvp-dataset/"
FILES = {
    "cosc.pth": (KAGGLE+urllib.parse.quote("AIR-OSVP-Dataset/OSVP-Checkpoint/best_ckpt.pth", safe="")+"?datasetVersionNumber=1", "best_ckpt.pth"),
    "cloud_0.txt": (KAGGLE+urllib.parse.quote("AIR-OSVP-Dataset/OSVP-Dataset/Asian_Dragon/rotate_0/cloud_0.txt", safe="")+"?datasetVersionNumber=1", "cloud_0.txt"),
    "poco.pth": ("https://github.com/valeoai/POCO/releases/download/v0.0.0/ShapeNet_3k.zip", "checkpoint.pth"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, (url, member_name) in FILES.items():
        target = args.output/name
        if not target.is_file():
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(url, timeout=180) as response:
                content = response.read()
            if name.endswith(".txt") and not zipfile.is_zipfile(io.BytesIO(content)):
                points = np.loadtxt(io.BytesIO(content), ndmin=2)
                if points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError("The public cloud must contain finite XYZ rows")
                target.write_bytes(content)
            else:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    members = [member for member in archive.infolist() if Path(member.filename).name == member_name]
                    if len(members) != 1:
                        raise ValueError(f"Expected one {member_name} in the public release")
                    target.write_bytes(archive.read(members[0]))
        metadata[name] = dict(source=url, bytes=target.stat().st_size)
    metadata["observation"] = "Asian_Dragon, rotate_0, original view 0 only; no future clouds fetched"
    metadata["kaggle_dataset_version"] = 1
    (args.output/"assets.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
