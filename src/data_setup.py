"""Qlib initialization and data download."""

import qlib
from pathlib import Path


def init_qlib(provider_uri: str = "./data/qlib_data_cn", region: str = "cn"):
    """Initialize Qlib with the given data directory and region."""
    provider_path = Path(provider_uri)
    qlib.init(provider_uri=str(provider_path.resolve()), region=region)
    print(f"Qlib initialized: provider_uri={provider_path.resolve()}, region={region}")


def download_data(target_dir: str = "./data/qlib_data_cn", region: str = "cn"):
    """Download stock data using Qlib's official get_data.py script.

    Only needs to be run once.
    """
    import subprocess
    import sys
    import urllib.request
    import os

    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()):
        print(f"Data already exists at {target_path.resolve()}, skipping download.")
        return

    target_path.mkdir(parents=True, exist_ok=True)
    
    script_url = "https://raw.githubusercontent.com/microsoft/qlib/main/scripts/get_data.py"
    script_path = "get_data.py"
    
    try:
        print(f"Downloading get_data.py from {script_url}...")
        urllib.request.urlretrieve(script_url, script_path)
        
        cmd = [
            sys.executable, script_path,
            "qlib_data",
            "--target_dir", str(target_path.resolve()),
            "--region", region,
        ]
        print(f"Running download script: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Data download complete.")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


if __name__ == "__main__":
    download_data()
