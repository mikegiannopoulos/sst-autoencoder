import os
from urllib.request import urlretrieve

def download_oisst_year(year="2024", save_dir="data"):
    url = f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"sst_{year}.nc")

    if not os.path.exists(out_path):
        print(f"📦 Downloading OISST for {year} ...")
        try:
            urlretrieve(url, out_path)
            print(f"✅ Downloaded to: {out_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
    else:
        print(f"✔ Already exists: {out_path}")

if __name__ == "__main__":
    download_oisst_year("2024")
