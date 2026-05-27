import torch
import torch.nn as nn

# Механізм відмовостійкого імпорту для локального тестування та глобального виконання
try:
    from kan_conv2d import KANConv2d
except ImportError:
    from src.kan_conv2d import KANConv2d

class UKANBlock(nn.Module):
    """
    Базовий обчислювальний блок архітектури U-KAN.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Примусове зниження роздільної здатності сплайнової сітки для економії VRAM
        self.kan = KANConv2d(in_channels, out_channels, kernel_size=3, padding=1, grid_size=3)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.kan(x))

class UKAN(nn.Module):
    """
    Повна архітектура U-KAN для семантичної сегментації мультимодальних геоданих.
    """
    def __init__(self, in_channels=15, num_classes=8):
        super().__init__()
        
        # Енкодер (Зниження просторової розмірності, збільшення глибини ознак)
        self.enc1 = UKANBlock(in_channels, 32)
        self.enc2 = UKANBlock(32, 64)
        self.enc3 = UKANBlock(64, 128)
        
        # Оператор субдискретизації
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Ботлнек (Найглибший рівень абстракції)
        self.bottleneck = UKANBlock(128, 256)
        
        # Декодер (Відновлення розмірності за допомогою транспонованих згорток)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Вхідний канал = 128 (з upconv) + 128 (з skip connection enc3) = 256
        self.dec3 = UKANBlock(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Вхідний канал = 64 (з upconv) + 64 (з enc2) = 128
        self.dec2 = UKANBlock(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Вхідний канал = 32 (з upconv) + 32 (з enc1) = 64
        self.dec1 = UKANBlock(64, 32)
        
        # Фінальний класифікатор (Лінійна проекція у простір 8 класів IGBP)
        # Використання класичної згортки 1x1 є стандартом для мінімізації FLOPs на виході
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Прохід енкодера
        e1 = self.enc1(x)       # Розмірність: [Batch, 32, 96, 96]
        p1 = self.pool(e1)      # Розмірність: [Batch, 32, 48, 48]
        
        e2 = self.enc2(p1)      # Розмірність: [Batch, 64, 48, 48]
        p2 = self.pool(e2)      # Розмірність: [Batch, 64, 24, 24]
        
        e3 = self.enc3(p2)      # Розмірність: [Batch, 128, 24, 24]
        p3 = self.pool(e3)      # Розмірність: [Batch, 128, 12, 12]
        
        # Прохід ботлнека
        b = self.bottleneck(p3) # Розмірність: [Batch, 256, 12, 12]
        
        # Прохід декодера (включає конкатенацію Skip Connections)
        d3 = self.upconv3(b)                                # [Batch, 128, 24, 24]
        d3 = torch.cat([d3, e3], dim=1)                     # [Batch, 256, 24, 24]
        d3 = self.dec3(d3)                                  # [Batch, 128, 24, 24]
        
        d2 = self.upconv2(d3)                               # [Batch, 64, 48, 48]
        d2 = torch.cat([d2, e2], dim=1)                     # [Batch, 128, 48, 48]
        d2 = self.dec2(d2)                                  # [Batch, 64, 48, 48]
        
        d1 = self.upconv1(d2)                               # [Batch, 32, 96, 96]
        d1 = torch.cat([d1, e1], dim=1)                     # [Batch, 64, 96, 96]
        d1 = self.dec1(d1)                                  # [Batch, 32, 96, 96]
        
        out = self.final_conv(d1)                           # [Batch, 8, 96, 96]
        return out

# Валідаційний блок
if __name__ == "__main__":
    print("Ініціалізація графа U-KAN...")
    model = UKAN(in_channels=15, num_classes=8)
    
    # Симуляція нормалізованого батчу DFC2020
    dummy_input = torch.randn(2, 15, 96, 96)
    
    print(f"Форма вхідного тензора: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Форма тензора передбачень (логітів): {output.shape}")
    
    # Розрахунок загальної кількості параметрів
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Загальна кількість навчальних параметрів моделі: {total_params:,}")