import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from dataloader import build_sevir
from model_ode import SimvpFlowODE

# "train" or "test"
MODE = "train"  

def save_uint8(arr, path):
    img = Image.fromarray(arr.astype(np.uint8), mode='L')
    img.save(path)


def run_inference(
    save_dir=f"ode_output/{MODE}",
    ckpt_path="checkpoints/best_model.pth",
    batch_size=4,
    num_workers=4,
    frames_per_sample=-1,
    seed=42,
):
    if frames_per_sample != -1 and frames_per_sample <= 0:
        raise ValueError("frames_per_sample must be -1 or a positive integer.")

    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dir_pred = os.path.join(save_dir, "pred") 
    dir_gt = os.path.join(save_dir, "gt")  
    os.makedirs(dir_pred, exist_ok=True)
    os.makedirs(dir_gt, exist_ok=True)

    model = SimvpFlowODE(shape_in=[12, 1, 384, 384]).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    ds_train, ds_test = build_sevir()
    ds = ds_train if MODE == "train" else ds_test
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
        )
    total_batches = len(loader)
    print(f"\n[{MODE}] {len(ds)} samples, {total_batches} batches")
    if frames_per_sample == -1:
        print("Saving all frames per sample\n")
    else:
        print(f"Randomly saving {frames_per_sample} frames per sample (seed={seed})\n")
    
    counter = 1

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.permute(0, 1, 4, 2, 3).to(device) 
            inputs, gts = data[:, 1:13], data[:, 13:]

            Y_fm_12, _ = model(inputs)

            pred_np = (Y_fm_12[:, :, 0] * 255.0).clamp(0, 255).byte().cpu().numpy()
            gt_np   = (gts[:, :, 0]     * 255.0).clamp(0, 255).byte().cpu().numpy()

            B, T_out, H, W = pred_np.shape

            for b in range(B):
                if frames_per_sample == -1 or frames_per_sample >= T_out:
                    T_seq = range(T_out)
                else:
                    T_seq = sorted(random.sample(range(T_out), frames_per_sample))
                for t in T_seq:
                    fname = f"{counter:07d}.png"
                    save_uint8(pred_np[b, t], os.path.join(dir_pred, fname))
                    save_uint8(gt_np[b, t],   os.path.join(dir_gt, fname))
                    counter += 1
                
            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == total_batches:
                print(f"  batch {batch_idx + 1}/{total_batches} | frames saved so far: {counter - 1}")

    print(f"\nFinished! Total frames saved: {counter - 1}")


if __name__ == "__main__":
    run_inference( 
        save_dir=f"ode_output/{MODE}",
        ckpt_path=f"checkpoints/best_model.pth",
        batch_size=12,
    )
