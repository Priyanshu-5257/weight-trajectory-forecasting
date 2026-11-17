import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from src.models import GrokkingTransformer, eval_grokking_tensors
from src.data import load_modular_addition_dataset

def run_baseline(p=97, frac_train=0.5, total_steps=12000, lr=1e-3, weight_decay=1.0, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Grokking Baseline Verification: p={p}, frac_train={frac_train}, device={device} ---")
    
    torch.manual_seed(seed)
    data = load_modular_addition_dataset(p=p, frac_train=frac_train, seed=seed, device=device)
    train_x, train_y = data["train_data"], data["train_labels"]
    test_x, test_y = data["test_data"], data["test_labels"]
    print(f"Dataset: Train={len(train_x)} ({frac_train*100:.0f}%), Test={len(test_x)}")

    model = GrokkingTransformer(p=p).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    history = []
    grokked_step = None

    for step in range(1, total_steps + 1):
        model.train()
        logits = model(train_x)
        loss = criterion(logits, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 250 == 0 or step == 1 or step == total_steps:
            train_loss, train_acc = eval_grokking_tensors(model, train_x, train_y, criterion)
            test_loss, test_acc = eval_grokking_tensors(model, test_x, test_y, criterion)
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"Step {step:5d}/{total_steps} [{elapsed:5.1f}s, {steps_per_sec:.0f} step/s] | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:5.1f}% | "
                  f"Test Loss: {test_loss:.4f}, Acc: {test_acc:5.1f}%")
            
            history.append({
                "step": step,
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc
            })

            if test_acc >= 95.0 and grokked_step is None:
                grokked_step = step
                print(f"🎉 GROKKED at step {step}! (Test Acc: {test_acc:.2f}%)")

    total_time = time.time() - t0
    print(f"Completed {total_steps} steps in {total_time:.1f}s ({total_steps/total_time:.0f} steps/sec).")
    if grokked_step:
        print(f"SUCCESS: Grokking confirmed at step {grokked_step}.")
    else:
        print("Note: Model did not reach 95% test accuracy within total_steps.")

if __name__ == "__main__":
    run_baseline(p=97, frac_train=0.5, total_steps=12000, seed=42)
