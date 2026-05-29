import os
import math
from datetime import datetime

# КРИТИЧНО ДЛЯ LINUX (4 ГБ VRAM): Запобігання фрагментації на рівні драйвера
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.dataset import DFC2020Dataset
from src.ukan_model import UKAN
from src.metrics import SegmentationMetrics

def train_pipeline():
    # Екстремальні параметри під жорсткий ліміт 4 ГБ VRAM
    MICRO_BATCH_SIZE = 1
    ACCUMULATION_STEPS = 16   # Ефективний розмір батчу = 16
    EPOCHS = 8                
    LEARNING_RATE = 2e-4      
    L1_LAMBDA = 1e-4          
    NUM_CLASSES = 8 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ПРИМУСОВЕ ВИМКНЕННЯ БЕНЧМАРКУ ДЛЯ ЗАХИСТУ ВІД ОЮМ
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ініціалізація завантаження датасетів (Пристрій: {device})...")
    full_train_dataset = DFC2020Dataset(split="train")
    full_val_dataset = DFC2020Dataset(split="val")
    
    # ---------------------------------------------------------
    # МОДУЛЬ СТАТИСТИЧНОЇ РЕДУКЦІЇ ДАТАСЕТУ (6%)
    # ---------------------------------------------------------
    generator = torch.Generator().manual_seed(42)
    
    train_total = len(full_train_dataset)
    train_target = int(train_total * 0.02) 
    train_remainder = train_total - train_target
    
    train_subset, _ = torch.utils.data.random_split(
        full_train_dataset, [train_target, train_remainder], generator=generator
    )
    
    val_total = len(full_val_dataset)
    val_target = int(val_total * 0.02)
    val_remainder = val_total - val_target
    
    val_subset, _ = torch.utils.data.random_split(
        full_val_dataset, [val_target, val_remainder], generator=generator
    )
    # ---------------------------------------------------------
    
    print(f"Редукована конфігурація: Train: {len(train_subset)} зразків | Val: {len(val_subset)} зразків")
    
    # МІНІМІЗАЦІЯ ПАМ'ЯТІ: num_workers=0 для повної ізоляції контексту
    train_loader = DataLoader(train_subset, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = UKAN(in_channels=15, num_classes=NUM_CLASSES).to(device)
    
    # Ініціалізація оптимізатора та планувальника ПІСЛЯ створення моделі
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2, fused=True)
    # ВИПРАВЛЕННЯ: Видалено застарілий аргумент verbose=True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    metrics_calc = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_miou = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    log_file = "checkpoints/training_log_2.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_miou,val_f1\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_valid_steps = 0
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # ЗАХИСТ ВІД NaN ПРИ НАКОПИЧЕННІ СТАТИСТИКИ
            loss_val = total_loss.item() * ACCUMULATION_STEPS
            if not math.isnan(loss_val):
                train_loss += loss_val
                train_valid_steps += 1
            
            if batch_idx % 100 == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Епоха [{epoch}/{EPOCHS}] | Ітерація [{batch_idx}/{len(train_loader)}] | Loss: {loss_val:.4f}")
        
        # Валідаційна фаза епохи
        model.eval()
        torch.cuda.empty_cache()
        val_loss = 0.0
        val_valid_steps = 0
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
                
                if not math.isnan(loss.item()):
                    val_loss += loss.item()
                    val_valid_steps += 1
                
                batch_metrics = metrics_calc.compute_batch_metrics(outputs, masks)
                if not math.isnan(batch_metrics["mIoU"]):
                    total_miou += batch_metrics["mIoU"]
                    total_f1 += batch_metrics["F1_Macro"]
                    valid_batches += 1
        
        avg_train_loss = (train_loss / train_valid_steps) if train_valid_steps > 0 else 0.0
        avg_val_loss = (val_loss / val_valid_steps) if val_valid_steps > 0 else 0.0
        avg_val_miou = (total_miou / valid_batches) if valid_batches > 0 else 0.0
        avg_val_f1 = (total_f1 / valid_batches) if valid_batches > 0 else 0.0
        
        epoch_end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{epoch_end_time}] [ЗВІТ ЕПОХИ {epoch}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val mIoU: {avg_val_miou:.4f} | Val F1-Macro: {avg_val_f1:.4f}\n")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_miou:.4f},{avg_val_f1:.4f}\n")
        
        # Оновлення планувальника швидкості навчання
        scheduler.step(avg_val_miou)
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Поточний Learning Rate: {current_lr:.6e}\n")
            
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save(model.state_dict(), "checkpoints/best_ukan_model_2.pth")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [!] Зафіксовано оптимум. Ваги збережено. (mIoU: {best_val_miou:.4f})\n")
            
if __name__ == "__main__":
    train_pipeline()