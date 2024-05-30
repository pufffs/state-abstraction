import torch

class SmoothLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        diff = x - y
        distance = 0.15*x.shape[1]
        root_norm = torch.sqrt(torch.sum(diff ** 2, dim=1))
        loss = torch.square(root_norm - distance)
        return loss.mean()
    
class CollapseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        diff = x - y
        root_norm = torch.sqrt(torch.sum(diff ** 2, dim=1))
        loss = torch.exp(-root_norm)
        return loss.mean()