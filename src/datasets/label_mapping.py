import numpy as np
from pathlib import Path
from PIL import Image

BACKGROUND,CROP,WEED=0,1,2
NUM_CLASS=3
NUM_CLASSES = NUM_CLASS
CLASS_NAMES=["Background","Crop","Weed"]
IGNORE_INDEX=255


_LUTS={
    "merge": np.array([0,1,2,1,2],dtype=np.uint8),
    "ignore":np.array([0,1,2,255,255],dtype=np.uint8),
}

def map_labels(mask:np.ndarray,mode:str="merge")->np.ndarray:
    mask=np.asarray(mask)
    if mask.ndim!=2:
        raise ValueError(f"Mask phải có 2 chiều, nhận{mask.shape}")
    if mask.max()>4:
        raise ValueError(f"Mask có ID ngoài 0-4: {np.unique(mask)}")
    if mode not in _LUTS:
        raise ValueError(f"mode phải là một trong {list(_LUTS)}")
    return _LUTS[mode][mask]


def _find_dataset_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    dataset = root / "data" / "raw" / "PhenoBench-v110" / "PhenoBench"
    if not dataset.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset tại: {dataset}")
    return dataset


if __name__=="__main__":
    dataset_root = _find_dataset_root()
    sem_dir = dataset_root / "train" / "semantics"
    mask_files = sorted(sem_dir.glob("*.png"))[:3]

    if not mask_files:
        raise FileNotFoundError(f"Không có file mask nào trong: {sem_dir}")

    print(f"Dataset root: {dataset_root}")
    for p in mask_files:
        raw = np.array(Image.open(p))
        mapped = map_labels(raw, mode="merge")
        print(f"{p.name}: raw_ids={np.unique(raw).tolist()} -> mapped_ids={np.unique(mapped).tolist()} | dtype={mapped.dtype}")

    print("label_mapping OK")