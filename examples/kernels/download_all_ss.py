import ssgetpy
from tqdm import tqdm

matrices = ssgetpy.search(limit=5000)

for matrix in tqdm(matrices, desc="Downloading Matrices"):
    try:
        matrix.download(format='MM', extract=True)
    except Exception as e:
        print(f"Error downloading matrix {matrix.name} in group {matrix.group}: {e}")
