import os
import urllib.request
import gzip
import shutil

# 1) Correct URL of the PubChem SDF chunk (first 500K CIDs)
URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/Compound_000000001_000500000.sdf.gz"

# 2) Local paths
#DATA_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_DIR = "/mnt/d/akarmark/data"
LOCAL_GZ  = os.path.join(DATA_DIR, "Compound_000000001_000500000.sdf.gz")
LOCAL_SDF = os.path.join(DATA_DIR, "Compound_000000001_000500000.sdf")

# 3) Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# 4) Download
print(f"Downloading {URL}\n  → {LOCAL_GZ} ...")
urllib.request.urlretrieve(URL, LOCAL_GZ)
print("Download complete.")

# 5) Decompress
print(f"Decompressing {LOCAL_GZ}\n  → {LOCAL_SDF} ...")
with gzip.open(LOCAL_GZ, "rb") as f_in, open(LOCAL_SDF, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
print("Decompression complete.")
