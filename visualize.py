import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Імпорт архітектури та датасету для інференсу
from src.dataset import DFC2020Dataset
from src.ukan_model import UKAN

# ==========================================
# ЧАСТИНА 1: АНАЛІЗ МЕТРИК ТРЕНУВАННЯ
# ==========================================
def plot_training_metrics(log_csv_path, return_fig=True):
    if not os.path.exists(log_csv_path):
        raise FileNotFoundError(f"Файл логу {log_csv_path} не знайдено.")
        
    df = pd.read_csv(log_csv_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Графік Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Епоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Динаміка помилки (Convergence)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Графік Метрик
    axes[1].plot(df['epoch'], df['val_miou'] * 100, label='Val mIoU (%)', marker='^')
    if 'val_f1' in df.columns:
        axes[1].plot(df['epoch'], df['val_f1'] * 100, label='Val F1-Macro (%)', marker='d')
    axes[1].set_xlabel('Епоха')
    axes[1].set_ylabel('Точність (%)')
    axes[1].set_title('Метрики сегментації')
    axes[1].legend()
    axes[1].grid(True)
    
    # Графік Learning Rate
    if 'lr' in df.columns:
        axes[2].plot(df['epoch'], df['lr'], label='Learning Rate', color='purple', marker='x')
        axes[2].set_yscale('log')
        axes[2].set_ylabel('Швидкість навчання (log scale)')
    else:
        axes[2].text(0.5, 0.5, 'Дані LR відсутні', ha='center', va='center')
    axes[2].set_xlabel('Епоха')
    axes[2].set_title('Траєкторія Планувальника (Scheduler)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.suptitle(f"Аналіз процесу тренування: {os.path.basename(log_csv_path)}", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if return_fig:
        plt.close(fig) # Запобігає подвійному рендерингу в Jupyter
        return fig
    else:
        plt.show()

# ==========================================
# ЧАСТИНА 2: ВІЗУАЛІЗАЦІЯ ІНФЕРЕНСУ (RGB)
# ==========================================
def get_dfc2020_cmap():
    """Створення кольорової палітри для 8 класів DFC2020 + фон"""
    colors = [
        '#228B22',  # 0: Ліс
        '#8B4513',  # 1: Чагарники
        '#ADFF2F',  # 2: Трава
        '#20B2AA',  # 3: Водно-болотні
        '#F4A460',  # 4: Сільгосп угіддя
        '#FF0000',  # 5: Забудова/Місто
        '#A9A9A9',  # 6: Пустир
        '#0000FF',  # 7: Вода
        '#000000'   # 8: Нерозмічено (Фон)
    ]
    return ListedColormap(colors)

def visualize_inference(checkpoint_path="checkpoints/best_ukan_model.pth", num_samples=3, norm_type="batch", return_fig=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ПЕРЕДАЧА ПАРАМЕТРА В АРХІТЕКТУРУ (Динамічна нормалізація)
    model = UKAN(in_channels=15, num_classes=8, norm_type=norm_type).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Файл ваг {checkpoint_path} не знайдено.")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    dataset = DFC2020Dataset(split="val")
    indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
    
    cmap = get_dfc2020_cmap()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, gt_mask = dataset[idx]
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            with torch.amp.autocast('cuda'):
                output = model(img_batch)
            
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            gt_mask = gt_mask.numpy()
            gt_mask_vis = np.where(gt_mask >= 8, 8, gt_mask)
            
            # Реконструкція Оптики (RGB)
            rgb_indices = [5, 4, 3]
            vis_img = img_tensor[rgb_indices].numpy()
            vis_img = np.transpose(vis_img, (1, 2, 0))
            
            p2, p98 = np.percentile(vis_img, (2, 98))
            vis_img = np.clip(vis_img, p2, p98)
            vis_img = (vis_img - p2) / (p98 - p2 + 1e-8)

            # Рендеринг
            axes[i][0].imshow(vis_img)
            axes[i][0].set_title(f"Оптика (RGB) (Патч #{idx})")
            axes[i][0].axis('off')

            axes[i][1].imshow(gt_mask_vis, cmap=cmap, vmin=0, vmax=8, interpolation='nearest')
            axes[i][1].set_title("Еталонна маска")
            axes[i][1].axis('off')

            axes[i][2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=8, interpolation='nearest')
            axes[i][2].set_title(f"Прогноз U-KAN ({norm_type})")
            axes[i][2].axis('off')

    plt.tight_layout()
    
    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()