import torch
import torch.nn as nn

class RegressionMLP(nn.Module):
    def __init__(self, in_features=8, hidden1=32, hidden2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class CompactMNISTMLP(nn.Module):
    def __init__(self, in_features=196, hidden_dim=16, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = nn.functional.adaptive_avg_pool2d(x, (14, 14)).view(x.size(0), -1)
        return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def eval_regression_mse(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        return criterion(preds, y_tensor).item()

def eval_mnist_tensors(model, x_tensor, y_tensor, criterion):
    model.eval()
    with torch.no_grad():
        out = model(x_tensor)
        loss = criterion(out, y_tensor).item()
        preds = out.argmax(dim=1)
        acc = 100.0 * (preds == y_tensor).sum().item() / y_tensor.size(0)
    return loss, acc
