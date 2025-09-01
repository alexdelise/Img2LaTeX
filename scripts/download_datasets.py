import kagglehub
import shutil
from pathlib import Path

# Image2Latex-100k Dataset
data_dir = Path("data/im2latex-100k")
data_dir.mkdir(parents=True, exist_ok=True)
cache_path = kagglehub.dataset_download("shahrukhkhan/im2latex100k")
print("Cache path:", cache_path)
shutil.copytree(cache_path, data_dir, dirs_exist_ok=True)
print("Dataset available at:", data_dir.resolve())
