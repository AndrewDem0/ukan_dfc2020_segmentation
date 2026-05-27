import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Імпорт ваших модулів
from src.dataset import DFC2020Dataset
from src.ukan_model import UKAN

def get_dfc2020_cmap():
    """Створення кольорової палітри для 8 класів DFC2020 + фон"""
    colors = [
        '#228B22',  # 0: Ліс (Зелений)
        '#8B4513',  # 1: Чагарники (Коричневий)
        '#ADFF2F',  # 2: Трава (Салатовий)
        '#20B2AA',  # 3: Водно-болотні (Морська хвиля)
        '#F4A460',  # 4: Сільгосп угіддя (Піщаний)
        '#FF0000',  # 5: Забудова/Місто (Червоний)
        '#A9A9A9',  # 6: Пустир (Сірий)
        '#0000FF',  # 7: Вода (Синій)
        '#000000'   # 8: Нерозмічено (Чорний для індексу 255)
    ]
    return ListedColormap(colors)

def visualize_inference(checkpoint_path="checkpoints/best_smoke_model.pth", num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ініціалізація інференсу на {device}...")

    # 1. Завантаження моделі
    model = UKAN(in_channels=15, num_classes=8).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Ваги успішно завантажено з: {checkpoint_path}")
    except Exception as e:
        print(f"Помилка завантаження ваг: {e}")
        return

    model.eval()
    dataset = DFC2020Dataset(split="val")
    
    # 2. Вибір випадкових патчів (виправлено конфлікт типів NumPy -> Python)
    indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
    
    cmap = get_dfc2020_cmap()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, gt_mask = dataset[idx]
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            # Прямий прохід
            with torch.amp.autocast('cuda'):
                output = model(img_batch)
            
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            gt_mask = gt_mask.numpy()
            gt_mask_vis = np.where(gt_mask >= 8, 8, gt_mask)
            
            # --- РЕКОНСТРУКЦІЯ ОПТИЧНОГО СПЕКТРА (RGB) ---
            # Витягуємо канали Sentinel-2: Red (5), Green (4), Blue (3)
            rgb_indices = [5, 4, 3]
            vis_img = img_tensor[rgb_indices].numpy() # [3, 96, 96]
            
            # Транспонування осей для Matplotlib: [H, W, Channels]
            vis_img = np.transpose(vis_img, (1, 2, 0))
            
            # Відновлення контрасту (відсікання 2% викидів після Z-нормалізації)
            p2, p98 = np.percentile(vis_img, (2, 98))
            vis_img = np.clip(vis_img, p2, p98)
            vis_img = (vis_img - p2) / (p98 - p2 + 1e-8)

            # Рендеринг колонок
            ax_img = axes[i][0]
            ax_img.imshow(vis_img)
            ax_img.set_title(f"Оптика (RGB) (Патч #{idx})")
            ax_img.axis('off')

            ax_gt = axes[i][1]
            ax_gt.imshow(gt_mask_vis, cmap=cmap, vmin=0, vmax=8, interpolation='nearest')
            ax_gt.set_title("Еталонна маска")
            ax_gt.axis('off')

            ax_pred = axes[i][2]
            ax_pred.imshow(pred_mask, cmap=cmap, vmin=0, vmax=8, interpolation='nearest')
            ax_pred.set_title("Прогноз U-KAN (Inference)")
            ax_pred.axis('off')

    plt.tight_layout()
    file_name = "inference_smoke_test_rgb.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Візуалізацію збережено у файл: {file_name}")
    plt.show()

if __name__ == "__main__":
    visualize_inference(checkpoint_path="checkpoints/best_smoke_model.pth")