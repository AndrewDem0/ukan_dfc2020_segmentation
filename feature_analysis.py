import torch
from src.ukan_model import UKAN

def analyze_channel_importance(checkpoint_path="checkpoints/best_ukan_model_1.pth"):
    device = torch.device("cpu")
    
    in_channels = 15
    num_classes = 8
    kernel_size = 3 
    
    # Ініціалізація архітектури
    model = UKAN(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # Завантаження ваг
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Вектор ваги успішно завантажено. Ініціалізація тензорного аналізу...\n")
    except FileNotFoundError:
        print(f"Критична помилка: Файл чекпойнту '{checkpoint_path}' не знайдено.")
        return

    # Динамічна ідентифікація та екстракція тензора сплайнів першого шару (L0)
    first_layer_splines = None
    layer_name_ref = ""
    
    for name, param in model.named_parameters():
        if 'spline_weight' in name:
            first_layer_splines = param
            layer_name_ref = name
            break  # Захоплення лише першого (вхідного) шару енкодера

    if first_layer_splines is None:
        print("Критична помилка: Тензор 'spline_weight' не знайдено у графі моделі.")
        return
        
    print(f"Цільовий тензор ідентифіковано: {layer_name_ref}\n")

    # Вихідна розмірність тензора: [out_channels, in_channels * K * K, grid_size + spline_order]
    out_c, in_k2, grid_k = first_layer_splines.shape
    
    # Верифікація розмірностей
    if in_k2 != in_channels * (kernel_size ** 2):
        print(f"Помилка розмірності. Очікувалося {in_channels * (kernel_size**2)}, отримано {in_k2}")
        return

    # Реструктуризація тензора для ізоляції вхідних каналів
    # Розмірність: [out_channels, in_channels, K*K, grid_size + spline_order]
    splines_reshaped = first_layer_splines.view(out_c, in_channels, kernel_size**2, grid_k)

    # Розрахунок L1-норми по всіх вимірах, крім осі вхідних каналів (dim=1)
    channel_importance_l1 = torch.norm(splines_reshaped, p=1, dim=(0, 2, 3))

    # Нормалізація у відсотковий розподіл
    total_importance = channel_importance_l1.sum()
    importance_percentage = (channel_importance_l1 / total_importance) * 100

    # Виведення результатів
    print("Апаратний аналіз значущості спектральних/радарних каналів (Рівень L0):")
    print("-" * 65)
    print(f"{'Канал':<10} | {'L1-Норма':<15} | {'Відносний внесок (%)':<20}")
    print("-" * 65)
    
    for ch in range(in_channels):
        norm_val = channel_importance_l1[ch].item()
        pct_val = importance_percentage[ch].item()
        print(f"Канал {ch:02d}   | {norm_val:<15.4f} | {pct_val:.2f}%")

if __name__ == "__main__":
    # Перевірте назву файлу чекпойнту (best_ukan_model.pth або best_ukan_model_1.pth)
    analyze_channel_importance(checkpoint_path="checkpoints/best_ukan_model_1.pth")