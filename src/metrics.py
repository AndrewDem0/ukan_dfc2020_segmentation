import torch
import numpy as np
from sklearn.metrics import jaccard_score, f1_score

class SegmentationMetrics:
    def __init__(self, num_classes: int = 8):
        self.num_classes = num_classes

    @torch.no_grad()
    def compute_batch_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict:
        preds = torch.argmax(outputs, dim=1)
        
        preds_flat = preds.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        valid_indices = targets_flat < self.num_classes
        preds_flat = preds_flat[valid_indices]
        targets_flat = targets_flat[valid_indices]
        
        # Блокування розрахунку для повністю нерозмічених патчів
        if len(targets_flat) == 0:
            return {
                "mIoU": float('nan'),
                "F1_Macro": float('nan')
            }
        
        labels = np.arange(self.num_classes)
        
        m_iou = jaccard_score(
            targets_flat, 
            preds_flat, 
            average='macro', 
            labels=labels, 
            zero_division=0
        )
        
        m_f1 = f1_score(
            targets_flat, 
            preds_flat, 
            average='macro', 
            labels=labels, 
            zero_division=0
        )
        
        return {
            "mIoU": float(m_iou),
            "F1_Macro": float(m_f1)
        }

if __name__ == "__main__":
    print("Тестування модуля метрик...")
    metrics_calculator = SegmentationMetrics(num_classes=8)
    dummy_outputs = torch.randn(2, 8, 96, 96)
    dummy_targets = torch.randint(0, 8, (2, 96, 96))
    res = metrics_calculator.compute_batch_metrics(dummy_outputs, dummy_targets)
    print(f" -> Mean IoU: {res['mIoU']:.4f}\n -> F1 Macro: {res['F1_Macro']:.4f}")