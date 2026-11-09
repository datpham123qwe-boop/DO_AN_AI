from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.label_mapping import map_labels


class PhenoBenchDataset(Dataset):
    def __init__(self, root=None, split="train", transform=None, mask_mode="merge"):
        if root is None:
            repo_root = Path(__file__).resolve().parents[2]
            root = repo_root / "data" / "raw" / "PhenoBench-v110" / "PhenoBench"
        else:
            root = Path(root)
            if not (root / split / "images").is_dir():
                candidate = root / "data" / "raw" / "PhenoBench-v110" / "PhenoBench"
                if (candidate / split / "images").is_dir():
                    root = candidate

        self.root=Path(root)
        self.split=split
        self.transform=transform
        self.mask_mode=mask_mode
        self.img_dir=self.root/split/"images"
        self.sem_dir=self.root/split/"semantics"
        self.files=sorted(p.name for p in self.img_dir.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"Không thấy ảnh trong {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name=self.files[i]
        image=np.array(Image.open(self.img_dir/name).convert("RGB"))
        mask=map_labels(np.array(Image.open(self.sem_dir/name)),self.mask_mode)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            mask = torch.from_numpy(mask.astype(np.int64, copy=False))

        return {"image": image, "mask": mask.long(), "filename": name}
