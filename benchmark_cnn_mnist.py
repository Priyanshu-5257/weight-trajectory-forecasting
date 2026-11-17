import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import TimesFmModelForPrediction

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 7 * 7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, 100.0 * correct / total

def run_cnn_benchmark(
    seeds=[42, 43],
    total_steps=1500,
    sat_step=1200,
    horizon=48,
    context_len=60,
    device_str="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    print("================================================================================")
    print(" CONVOLUTIONAL ARCHITECTURE REPLICATION (SMALL CNN ON MNIST)")
    print(f" Seeds: {seeds} | Total Steps: {total_steps} | Cut-off: Step {sat_step}")
    print("================================================================================")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    tfm_model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    tfm_model.eval()

    def forecast_weights(past_w_np, h):
        P = past_w_np.shape[0]
        tensor_in = torch.tensor(past_w_np, dtype=torch.float32)
        out_list = []
        with torch.no_grad():
            for b_start in range(0, P, 1024):
                b_end = min(b_start + 1024, P)
                out = tfm_model(past_values=tensor_in[b_start:b_end].to(device))
                out_list.append(out.mean_predictions[:, :h].cpu().numpy())
        return np.concatenate(out_list, axis=0)

    cnn_results = []
    for seed in seeds:
        print(f"\n--- Training CNN Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        model = SmallCNN().to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"CNN Parameter Count: {num_params}")

        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        weight_history = []
        weight_history.append(get_flat_params(model).cpu())

        data_iter = iter(train_loader)
        for step in range(1, total_steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            weight_history.append(get_flat_params(model).cpu())

        # Evaluate at Saturation
        w_frozen = weight_history[sat_step]
        set_flat_params(model, w_frozen.to(device))
        l_froz, a_froz = evaluate(model, test_loader, device, criterion)

        # Ground truth continuation
        w_gt = weight_history[sat_step + horizon]
        set_flat_params(model, w_gt.to(device))
        l_gt, a_gt = evaluate(model, test_loader, device, criterion)

        # Matched SWA
        s_start = sat_step - context_len + 1
        W_sat = torch.stack(weight_history[s_start:sat_step + 1], dim=0).numpy().T
        w_swa = torch.mean(torch.stack(weight_history[s_start:sat_step + 1], dim=0), dim=0)
        set_flat_params(model, w_swa.to(device))
        l_swa, a_swa = evaluate(model, test_loader, device, criterion)

        # TimesFM forecast
        print(f"Forecasting {num_params} CNN parameters with TimesFM...")
        t0 = time.time()
        w_tfm = torch.tensor(forecast_weights(W_sat, horizon)[:, -1], dtype=torch.float32)
        print(f"CNN forecast completed in {time.time() - t0:.2f}s.")
        set_flat_params(model, w_tfm.to(device))
        l_tfm, a_tfm = evaluate(model, test_loader, device, criterion)

        print(f"Seed {seed} CNN Results: Frozen={l_froz:.4f} ({a_froz:.2f}%) | SWA={l_swa:.4f} ({a_swa:.2f}%) | TimesFM={l_tfm:.4f} ({a_tfm:.2f}%) | More SGD={l_gt:.4f} ({a_gt:.2f}%)")
        cnn_results.append({
            "seed": seed,
            "frozen": {"loss": l_froz, "acc": a_froz},
            "gt_sgd": {"loss": l_gt, "acc": a_gt},
            "swa": {"loss": l_swa, "acc": a_swa},
            "timesfm": {"loss": l_tfm, "acc": a_tfm}
        })

    with open("cnn_replication_results.json", "w") as f:
        json.dump(cnn_results, f, indent=2)
    print("\nSaved CNN results to cnn_replication_results.json")

    return cnn_results

if __name__ == "__main__":
    run_cnn_benchmark(seeds=[42, 43])
