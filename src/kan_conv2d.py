import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """
    Базовий 1D KAN-шар. 
    Апроксимує функцію на ребрах графа через комбінацію SiLU та B-сплайнів.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Базова вагова матриця (для функції активації SiLU)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Вагова матриця для B-сплайнів
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # Сітка контрольних точок (Grid)
        grid = torch.linspace(-1, 1, steps=grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def b_splines(self, x):
        """Обчислення базисних функцій B-сплайна."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
                        # Замінити 1e-8 на 1e-4
            left_term = (x - self.grid[:-k - 1]) / (self.grid[k:-1] - self.grid[:-k - 1] + 1e-4) * bases[..., :-1]
            right_term = (self.grid[k + 1:] - x) / (self.grid[k + 1:] - self.grid[1:-k] + 1e-4) * bases[..., 1:]
            bases = left_term + right_term
            
        return bases

    def forward(self, x):
        # Базова активація: w * SiLU(x)
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # Сплайнова активація
        spline_basis = self.b_splines(x)  # [batch_size, in_features, grid_size + spline_order]
        # Скалярний добуток сплайнів та вагових коефіцієнтів
        spline_output = torch.einsum('bik,oik->bo', spline_basis, self.spline_weight)
        
        return base_output + spline_output


class KANConv2d(nn.Module):
    """
    2D Згортковий KAN-шар для обробки зображень/тензорів DFC2020.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, grid_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Обчислення вхідної розмірності для локального вікна згортки
        self.in_features = in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        # Ініціалізація лінійного KAN-шару для обробки вікон
        self.kan_linear = KANLinear(self.in_features, out_channels, grid_size=grid_size)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        
        # Розрахунок розмірності вихідного тензора
        out_h = (height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Екстракція локальних вікон (патчів)
        # Розмірність: [batch_size, in_features, num_patches]
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Транспонування для KANLinear: [batch_size * num_patches, in_features]
        x_unfolded = x_unfolded.transpose(1, 2).reshape(-1, self.in_features)
        
        # Застосування нейромережі Колмогорова-Арнольда до кожного вікна
        out_unfolded = self.kan_linear(x_unfolded)
        
        # Зворотне перетворення (Folding) у 2D тензор
        out = out_unfolded.view(batch_size, out_h * out_w, self.out_channels)
        out = out.transpose(1, 2).view(batch_size, self.out_channels, out_h, out_w)
        
        return out

# Тестовий блок
if __name__ == "__main__":
    print("Тестування KANConv2d...")
    # Симуляція тензора DFC2020 (batch_size=2, channels=15, 96x96)
    dummy_x = torch.randn(2, 15, 96, 96)
    kan_conv = KANConv2d(in_channels=15, out_channels=32, kernel_size=3, padding=1)
    
    out = kan_conv(dummy_x)
    print(f"Вхідний тензор: {dummy_x.shape}")
    print(f"Вихідний тензор після KAN-згортки: {out.shape}")