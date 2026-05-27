import os
# Системне налаштування розподільника пам'яті CUDA (повинно виконуватися до імпорту torch)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Імпорт компонентів ядра архітектури
from src.dataset import DFC2020Dataset
from src.ukan_model import UKAN
from src.metrics import SegmentationMetrics

def train_pipeline():
    # 1. Фізичні та логічні гіперпараметри для обмеженого VRAM (GTX 1650 4GB)
    MICRO_BATCH_SIZE = 1     # Фізичний обсяг тензора в пам'яті GPU
    ACCUMULATION_STEPS = 16  # Кількість кроків накопичення градієнтів (Віртуальний батч = 16)
    EPOCHS = 6
    LEARNING_RATE = 1e-3
    L1_LAMBDA = 1e-4
    NUM_CLASSES = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Апаратне середовище обчислень: {device}")
    
    # 2. Ініціалізація потоків даних
    print("Завантаження датасетів з системного кешу...")
    train_dataset = DFC2020Dataset(split="train")
    val_dataset = DFC2020Dataset(split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Компільованого графа та оптимізаторів
    model = UKAN(in_channels=15, num_classes=NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # Ігнорування індексу 255 під час розрахунку втрат та градієнтів
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    metrics_calc = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    # Стандартизований інтерфейс змішаної точності AMP
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_miou = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    log_file = "checkpoints/training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_miou,val_f1\n")
    print("Старт процесу навчання моделі U-KAN...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Скидання градієнтів перед ітераційним спуском епохи
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Фільтрація та перекодування неанотованих супутникових пікселів
            masks[masks >= NUM_CLASSES] = 255
            
            # Прямий проход в ізольованому контексті AMP (float16 обчислення)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Обчислення L1-норми для затухання неінформативних B-сплайнів KAN
                l1_reg = torch.tensor(0., device=device)
                for name, param in model.named_parameters():
                    if 'spline_weight' in name:
                        l1_reg += torch.norm(param, p=1)
                
                # Нормалізація втрат на коефіцієнт акумуляції
                total_loss = (loss + L1_LAMBDA * l1_reg) / ACCUMULATION_STEPS
            
            # Масштабований зворотний прохід
            scaler.scale(total_loss).backward()
            
            # Оновлення вагових коефіцієнтів моделі
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += (total_loss.item() * ACCUMULATION_STEPS)
            
            if batch_idx % 100 == 0:
                current_loss = total_loss.item() * ACCUMULATION_STEPS
                print(f"Епоха [{epoch}/{EPOCHS}] | Ітерація [{batch_idx}/{len(train_loader)}] | Loss: {current_loss:.4f}")
        
        # Валідаційна фаза епохи
        model.eval()
        torch.cuda.empty_cache()  # Звільнення фрагментованого кешу VRAM перед інференсом
        val_loss = 0.0
        total_miou = 0.0
        total_f1 = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks[masks >= NUM_CLASSES] = 255
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                batch_metrics = metrics_calc.compute_batch_metrics(outputs, masks)
                total_miou += batch_metrics["mIoU"]
                total_f1 += batch_metrics["F1_Macro"]
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_miou = total_miou / len(val_loader)
        avg_val_f1 = total_f1 / len(val_loader)
        
        print(f"\n[ЗВІТ ЕПОХИ {epoch}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val mIoU: {avg_val_miou:.4f} | Val F1-Macro: {avg_val_f1:.4f}\n")
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_miou:.4f},{avg_val_f1:.4f}\n")
            
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save(model.state_dict(), "checkpoints/best_ukan_model.pth")
            print(f"[!] Зафіксовано оптимум. Ваги збережено. (mIoU: {best_val_miou:.4f})\n")

if __name__ == "__main__":
    train_pipeline()