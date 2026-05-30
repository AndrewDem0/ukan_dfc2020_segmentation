import os
import math

# Примусове налаштування середовища Windows для диска E:
os.environ["HF_HOME"] = r"E:\.hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\.hf_cache\datasets"
os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.dataset import DFC2020Dataset
from src.ukan_model import UKAN
from src.metrics import SegmentationMetrics

def train_pipeline():
    # Гіперпараметри адаптовано під ліміти 6 ГБ VRAM
    MICRO_BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8
    EPOCHS = 6
    LEARNING_RATE = 1e-4
    L1_LAMBDA = 1e-4
    NUM_CLASSES = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Апаратне прискорення CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    print(f"Ініціалізація завантаження датасетів (Пристрій: {device})...")
    full_train_dataset = DFC2020Dataset(split="train")
    full_val_dataset = DFC2020Dataset(split="val")
    
    # ---------------------------------------------------------
    # МОДУЛЬ СТАТИСТИЧНОЇ РЕДУКЦІЇ ДАТАСЕТУ (25%)
    # ---------------------------------------------------------
    generator = torch.Generator().manual_seed(42)
    
    # Розрахунок розмірів для Train (25% / 75%)
    train_total = len(full_train_dataset)
    train_25 = int(train_total * 0.25)
    train_75 = train_total - train_25
    
    train_subset, _ = torch.utils.data.random_split(
        full_train_dataset, 
        [train_25, train_75], 
        generator=generator
    )
    
    # Розрахунок розмірів для Val (25% / 75%)
    val_total = len(full_val_dataset)
    val_25 = int(val_total * 0.25)
    val_75 = val_total - val_25
    
    val_subset, _ = torch.utils.data.random_split(
        full_val_dataset, 
        [val_25, val_75], 
        generator=generator
    )
    # ---------------------------------------------------------
    
    print(f"Редукована конфігурація: Train: {len(train_subset)} зразків | Val: {len(val_subset)} зразків")
    
    # КРИТИЧНО ДЛЯ WINDOWS: num_workers=0
    train_loader = DataLoader(
        train_subset, 
        batch_size=MICRO_BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=MICRO_BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    model = UKAN(in_channels=15, num_classes=NUM_CLASSES).to(device)
    
    # Оптимізація пам'яті: fused=True
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2, fused=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    metrics_calc = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_miou = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    log_file = "checkpoints/training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_miou,val_f1\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            masks[masks >= NUM_CLASSES] = 255
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                l1_reg = torch.tensor(0., device=device)
                for name, param in model.named_parameters():
                    if 'spline_weight' in name:
                        l1_reg += torch.norm(param, p=1)
                
                total_loss = (loss + L1_LAMBDA * l1_reg) / ACCUMULATION_STEPS
            
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # АНТИ-NaN ЗАХИСТ (Кліппінг градієнтів)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += (total_loss.item() * ACCUMULATION_STEPS)
            
            if batch_idx % 100 == 0:
                current_loss = total_loss.item() * ACCUMULATION_STEPS
                print(f"Епоха [{epoch}/{EPOCHS}] | Ітерація [{batch_idx}/{len(train_loader)}] | Loss: {current_loss:.4f}")
        
        # Валідаційна фаза епохи
        model.eval()
        torch.cuda.empty_cache()
        val_loss = 0.0
        total_miou = 0.0
        total_f1 = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks[masks >= NUM_CLASSES] = 255
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                batch_metrics = metrics_calc.compute_batch_metrics(outputs, masks)
                
                if not math.isnan(batch_metrics["mIoU"]):
                    total_miou += batch_metrics["mIoU"]
                    total_f1 += batch_metrics["F1_Macro"]
                    valid_batches += 1
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        avg_val_miou = (total_miou / valid_batches) if valid_batches > 0 else 0.0
        avg_val_f1 = (total_f1 / valid_batches) if valid_batches > 0 else 0.0
        
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