import os
import sys

# КРИТИЧНО: Глобальне перенаправлення кешу на диск E: (виконується до імпорту torch/datasets)
os.environ["HF_HOME"] = r"E:\.hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\.hf_cache\datasets"
os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DownloadConfig
import numpy as np

class DFC2020Dataset(Dataset):
    """
    Клас-обгортка для мультимодального датасету DFC2020 (Sentinel-1 + Sentinel-2).
    Забезпечує конкатенацію спектральних каналів та тензорну нормалізацію з локалізацією на диску E:.
    """
    def __init__(self, split: str = "train", transform=None):
        self.transform = transform
        self.num_channels = 15
        self.num_classes = 8
        
        # Фіксація абсолютної траєкторії на диску E:
        self.cache_dir = r"E:\.hf_cache\datasets"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        download_config = DownloadConfig(
            max_retries=15,
            resume_download=True
        )

        self.dataset = load_dataset(
            "GFM-Bench/DFC2020", 
            split=split, 
            trust_remote_code=True,
            download_config=download_config,
            cache_dir=self.cache_dir,
            num_proc=1
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        std = np.std(image, axis=(1, 2), keepdims=True)
        eps = 1e-8
        return (image - mean) / (std + eps)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.dataset[idx]
        s1_data = np.array(sample['radar'], dtype=np.float32)  
        s2_data = np.array(sample['optical'], dtype=np.float32)
        label = np.array(sample['label'], dtype=np.int64)   
        
        image = np.concatenate([s1_data, s2_data], axis=0)
        image = self._normalize_image(image)
        
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(label)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, mask_tensor

if __name__ == "__main__":
    print("Ініціалізація ізольованого завантаження на диск E:...")
    try:
        dummy_dataset = DFC2020Dataset(split="train") 
        img, msk = dummy_dataset[0]
        print(f"Успішно. Розмірність вхідного тензора (X): {img.shape}")
        print(f"Розмірність тензора маски (Y): {msk.shape}")
        print(f"Унікальні класи в масці: {torch.unique(msk)}")
    except Exception as e:
        print(f"Помилка ініціалізації датасету: {e}")