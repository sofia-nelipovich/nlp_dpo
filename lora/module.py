import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Реализация LoRA (Low-Rank Adaptation) слоя.
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        """
        original_linear: исходный nn.Linear слой из LLM, который будем адаптировать
        r: ранг низкорангового разложения
        alpha: коэффициент масштабирования
        """
        super().__init__()
        self.in_channels = original_linear.weight.shape[1]
        self.out_channels = original_linear.weight.shape[0]
        self.r = r
        self.alpha = alpha

        # Сохраняем исходные замороженные веса
        self.weight = original_linear.weight.detach().clone()
        self.bias = original_linear.bias.detach().clone() if original_linear.bias is not None else None

        # Создаём обучаемые матрицы B и A для low-rank адаптации
        # B: (out_channels, r) -- инициализируется нулями
        self.B = nn.Parameter(torch.zeros(self.out_channels, r, dtype=torch.float32))
        # A: (r, in_channels) -- инициализируется случайно, т.к. градиенты для A пойдут сразу
        self.A = nn.Parameter(torch.randn(r, self.in_channels, dtype=torch.float32) * 0.01) # N(0, 0.01)

        # print('B init:', self.B.mean(), self.B.std())
        # print('A init:', self.A.mean(), self.A.std())



    def forward(self, x):
        """
        Сначала считаем стандартный выход (замороженный), потом добавляем low-rank слагаемое BA.
        h = Wx + BAx, где W - pretrained, B и A - обучаемые.
        """
        # print(x.dtype, self.B.dtype, self.A.dtype)
        if torch.isnan(x).any():
            print("NaN in input x!")
        lora_mat = torch.matmul(self.B, self.A)
        if torch.isnan(lora_mat).any() or torch.isinf(lora_mat).any():
            print("NaN or Inf in lora_mat", lora_mat)
        # Основной замороженный вывод
        # print(x.shape, self.weight.shape, self.bias.shape)
        if x.shape[1] != self.weight.shape[0]:
            out = torch.nn.functional.linear(x, self.weight, self.bias)
            lora_out = torch.nn.functional.linear(x, lora_mat) * (self.alpha / self.r)
        else:
            out = torch.nn.functional.linear(x, self.weight.T, self.bias)
            lora_out = torch.nn.functional.linear(x, lora_mat.T) * (self.alpha / self.r)
        # LoRA-компонента
        return out + lora_out

    def num_trainable_parameters(self):
        """
        Возвращает количество обучаемых параметров (только матрицы B и A).
        """
        total = sum(p.numel() for p in [self.B, self.A])
        return total

    def num_total_parameters(self):
        """
        Возвращает общее число параметров (включая замороженные — для сравнения).
        """
        base = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
        lora = self.num_trainable_parameters()
        return base + lora
