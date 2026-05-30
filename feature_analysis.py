import os
import torch
import matplotlib.pyplot as plt
from src.ukan_model import UKAN

def analyze_channel_importance(checkpoint_path="checkpoints/best_ukan_model.pth", return_fig=True):
    device = torch.device("cpu")
    in_channels = 15
    num_classes = 8
    kernel_size = 3 
    
    model = UKAN(in_channels=in_channels, num_classes=num_classes).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпойнт {checkpoint_path} відсутній.")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    first_layer_splines = None
    for name, param in model.named_parameters():
        if 'spline_weight' in name:
            first_layer_splines = param
            break

    if first_layer_splines is None:
        raise ValueError("Тензор 'spline_weight' не знайдено в структурі графа.")

    out_c, in_k2, grid_k = first_layer_splines.shape
    splines_reshaped = first_layer_splines.view(out_c, in_channels, kernel_size**2, grid_k)
    channel_importance_l1 = torch.norm(splines_reshaped, p=1, dim=(0, 2, 3))
    importance_percentage = (channel_importance_l1 / channel_importance_l1.sum()) * 100

    # Побудова графіку
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = [f"Канал {i:02d}" for i in range(in_channels)]
    y_pos = range(in_channels)
    
    bars = ax.barh(y_pos, importance_percentage.detach().numpy(), align='center', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(channels)
    ax.invert_yaxis()  # Топ-канали зверху
    ax.set_xlabel('Відносний внесок (%)')
    ax.set_title(f'Аналіз значущості ознак (White-Box KAN L0)\nЧекпойнт: {os.path.basename(checkpoint_path)}')
    
    # Додавання текстових значень на графік
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                va='center', ha='left', fontsize=9)
                
    plt.tight_layout()
    
    if return_fig:
        plt.close(fig) # Запобігання дублюванню в Jupyter
        return fig
    else:
        plt.show()