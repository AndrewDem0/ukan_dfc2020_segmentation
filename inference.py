import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import DFC2020Dataset
from src.ukan_model import UKAN
from utils import seed_everything, ensure_dir

def run_production_inference():
    # 1. Системна ініціалізація
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/best_ukan_model.pth"
    output_dir = "inference_results"
    ensure_dir(output_dir)

    print(f"Ініціалізація конвеєра інференсу на пристрої: {device}")

    # 2. Завантаження макроструктури графа та ваг
    model = UKAN(in_channels=15, num_classes=8).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Файл ваг {checkpoint_path} не знайдено. Завершіть навчання.")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Ініціалізація потоку даних
    val_dataset = DFC2020Dataset(split="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Початок екстракції масок для {len(val_dataset)} патчів...")

    # 4. Прямий прохід із блокуванням градієнтів
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(val_loader, desc="Inference")):
            images = images.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            
            # Застосування Argmax для визначення домінантного класу (0-7)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Збереження бінарної матриці на диск
            np.save(os.path.join(output_dir, f"patch_{idx}_pred.npy"), pred_mask)

    print(f"Інференс завершено. Матриці класів збережено у директорію: {output_dir}/")

if __name__ == "__main__":
    run_production_inference()