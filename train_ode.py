import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import build_sevir
import eval_metrics as utpp
from model_ode import SimvpFlowODE


CSI_THRESHOLDS = [16, 74, 133, 160, 181, 219]


def evaluate_csi(preds, gts):
    """preds, gts: [B, T, 1, H, W]  (0~1)"""
    preds = preds.clamp(0, 1)
    return {
        threshold: np.array(
            utpp.tfpn(preds, gts, threshold=threshold / 255.0),
            dtype=np.float64,
        )
        for threshold in CSI_THRESHOLDS
    }


def compute_csi_m(csi_sums):
    return np.mean([utpp.csi(*csi_sums[threshold]) for threshold in CSI_THRESHOLDS])


# -----------------------------
# Setup
# -----------------------------
os.makedirs("checkpoints", exist_ok=True)

ds_train, ds_test = build_sevir()

train_loader = DataLoader(
    ds_train, batch_size=4, shuffle=True,
    num_workers=10, pin_memory=True
)
test_loader = DataLoader(
    ds_test, batch_size=24, shuffle=False,
    num_workers=10, pin_memory=True
)

device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 100

model = SimvpFlowODE(shape_in=[12, 1, 384, 384]).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6,
)

criterion = nn.MSELoss()
best_csi_m = -float("inf")


for epoch in range(num_epochs):

    # ===== Train =====
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
    for batch_idx, data in enumerate(pbar):
        data = data.permute(0, 1, 4, 2, 3).to(device) 
        inputs, gts = data[:, 1:13], data[:, 13:]

        optimizer.zero_grad()
        Y_teacher, Y_ode = model(inputs)

        L_pred = criterion(Y_teacher, gts)
        L_ode = criterion(Y_teacher, Y_ode.detach())
        loss = L_pred + 0.1 * L_ode
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    train_loss /= len(train_loader)
    scheduler.step()

    # ===== Validation =====
    model.eval()
    val_loss = 0.0
    csi_sums = {
        threshold: np.zeros(4, dtype=np.float64)
        for threshold in CSI_THRESHOLDS
    }

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val")
        for batch_idx, data in enumerate(pbar):
            data = data.permute(0, 1, 4, 2, 3).to(device)
            inputs, gts = data[:, 1:13], data[:, 13:]

            preds, _ = model(inputs)
            # preds = preds.clamp(0, 1)       

            loss      = criterion(preds, gts)
            val_loss += loss.item()

            csi = evaluate_csi(preds, gts)
            for threshold in CSI_THRESHOLDS:
                csi_sums[threshold] += csi[threshold]

            pbar.set_postfix(
                loss=f"{loss.item():.6f}",
                csi16=f"{utpp.csi(*csi[16]):.3f}"
            )

    val_loss /= len(test_loader)
    val_csi_m = compute_csi_m(csi_sums)

    print(
        f"\nEpoch [{epoch+1}/{num_epochs}] "
        f"Train {train_loss:.6f} | Val {val_loss:.6f}\n"
        f"CSI-M   : {val_csi_m:.4f} | "
        f"CSI@16  : {utpp.csi(*csi_sums[16]):.4f} | "
        f"CSI@160 : {utpp.csi(*csi_sums[160]):.4f} | "
        f"CSI@219 : {utpp.csi(*csi_sums[219]):.4f}\n"
    )

    if val_csi_m > best_csi_m:
        best_csi_m = val_csi_m
        torch.save(model.state_dict(), f"checkpoints/best_model.pth")   # fixed: no .module
        print(f"  --> Saved best model (CSI-M={best_csi_m:.6f})")
