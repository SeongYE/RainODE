import os
import sys
import argparse
import yaml
import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset

SOURCE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOURCE_MODEL_DIR)

from utils import dict2namespace
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.base.EMA import EMA


class AlignedDataset(Dataset):
    def __init__(self, input_dir, gt_dir, image_size):
        self.input_dir = input_dir
        self.gt_dir    = gt_dir
        self.filenames = sorted([f for f in os.listdir(input_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname    = self.filenames[idx]
        img_cond = Image.open(os.path.join(self.input_dir, fname)).convert('RGB')
        img_gt   = Image.open(os.path.join(self.gt_dir,    fname)).convert('RGB')
        return self.transform(img_cond), self.transform(img_gt), fname


def load_config(config_path):
    with open(config_path, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2namespace(dict_config)
    # Fix relative VQGAN ckpt path to absolute
    cfg.model.VQGAN.params.ckpt_path = os.path.join(SOURCE_MODEL_DIR,
                                                     cfg.model.VQGAN.params.ckpt_path.lstrip('./'))
    return cfg


def load_model(config, model_path, device):
    net = LatentBrownianBridgeModel(config.model).to(device)
    states = torch.load(model_path, map_location='cpu')
    net.load_state_dict(states['model'])
    if config.model.EMA.use_ema and 'ema' in states:
        ema = EMA(config.model.EMA.ema_decay)
        ema.register(net)
        ema.shadow = states['ema']
        ema.reset_device(net)
        ema.apply_shadow(net)
    net.eval()
    return net


def worker(rank, gpu_id, indices, args, config, shared, lock):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"[GPU {gpu_id}] Loading model...", flush=True)
    net = load_model(config, args.model, device)

    image_size = config.data.dataset_config.image_size
    to_normal  = config.data.dataset_config.to_normal

    input_dir = os.path.join(args.data_root, 'pred')
    gt_dir    = os.path.join(args.data_root, 'gt')

    dataset = AlignedDataset(input_dir, gt_dir, image_size)
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=args.batch_size,
                         num_workers=4, pin_memory=True, drop_last=False)

    pred_dir = os.path.join(args.output_dir, 'pred')
    gt_dir_out = os.path.join(args.output_dir, 'gt')

    csi_thresholds = [16, 160, 219]

    for x_cond, x_gt, fnames in loader:
        # Normalize for model
        if to_normal:
            x_cond_in = (x_cond - 0.5) * 2.0
        else:
            x_cond_in = x_cond
        x_cond_in = x_cond_in.to(device)

        with torch.no_grad():
            sample = net.sample(x_cond_in, clip_denoised=False)

        # Denormalize to [0, 255]
        pred_f = sample.float()
        if to_normal:
            pred_f = pred_f.mul(0.5).add(0.5).clamp(0, 1.)
        pred_255 = pred_f.mul(255).clamp(0, 255).cpu().numpy()   # [B, C, H, W]
        gt_255   = x_gt.float().mul(255).clamp(0, 255).numpy()   # [B, C, H, W]

        # Metrics
        batch_mse  = float(np.mean((pred_255 - gt_255) ** 2, axis=(1, 2, 3)).sum())
        pred_gray  = pred_255.mean(axis=1)   # [B, H, W]
        gt_gray    = gt_255.mean(axis=1)

        batch_csi = {}
        for t in csi_thresholds:
            pb = pred_gray >= t
            gb = gt_gray   >= t
            batch_csi[t] = (
                int(np.logical_and( pb,  gb).sum()),
                int(np.logical_and( pb, ~gb).sum()),
                int(np.logical_and(~pb,  gb).sum()),
            )

        # Save images
        for i, fname in enumerate(fnames):
            pred_img = pred_255[i].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
            gt_img   = gt_255[i].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
            Image.fromarray(pred_img).save(os.path.join(pred_dir,   fname))
            Image.fromarray(gt_img  ).save(os.path.join(gt_dir_out, fname))

        # Update shared accumulators and print
        with lock:
            shared['count']   += len(fnames)
            shared['mse_sum'] += batch_mse
            for t in csi_thresholds:
                TP, FP, FN = batch_csi[t]
                shared[f'TP_{t}'] += TP
                shared[f'FP_{t}'] += FP
                shared[f'FN_{t}'] += FN

            # Real-time accumulated metrics
            count   = shared['count']
            avg_mse = shared['mse_sum'] / count
            csi_parts = []
            for t in csi_thresholds:
                TP_a = shared[f'TP_{t}']
                FP_a = shared[f'FP_{t}']
                FN_a = shared[f'FN_{t}']
                den  = TP_a + FP_a + FN_a
                csi  = TP_a / den if den > 0 else 1.0
                csi_parts.append(f"CSI@{t}={csi:.4f}")

            print(f"[GPU{gpu_id}] [{count:>6}/{shared['total']}]  "
                  f"MSE={avg_mse:.4f}  " + "  ".join(csi_parts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='./configs/Template-LBBDM-f4.yaml')
    parser.add_argument('--model',      type=str, default='./results/sevir/LBBDM-f4/checkpoint/last_model.pth')
    parser.add_argument('--data_root',  type=str, default='../ode_output/test')
    parser.add_argument('--output_dir', type=str, default='../bbdm_output')
    parser.add_argument('--gpu_ids',    type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=12)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    config  = load_config(args.config)

    input_dir = os.path.join(args.data_root, 'pred')
    all_files = sorted([f for f in os.listdir(input_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total = len(all_files)

    os.makedirs(os.path.join(args.output_dir, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'gt'),   exist_ok=True)

    # Split indices across GPUs
    splits = [list(range(i, total, len(gpu_ids))) for i in range(len(gpu_ids))]

    # Shared accumulators
    manager = mp.Manager()
    shared  = manager.dict({
        'total': total, 'count': 0, 'mse_sum': 0.0,
        **{f'TP_{t}': 0 for t in [16, 160, 219]},
        **{f'FP_{t}': 0 for t in [16, 160, 219]},
        **{f'FN_{t}': 0 for t in [16, 160, 219]},
    })
    lock = manager.Lock()

    print(f"Evaluating {total} images | GPUs={gpu_ids} | batch_size={args.batch_size}")
    print(f"Saving pred → {args.output_dir}/pred/   GT → {args.output_dir}/gt/\n")

    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=worker,
                       args=(rank, gpu_id, splits[rank], args, config, shared, lock))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Final summary
    csi_thresholds = [16, 160, 219]
    count = shared['count']
    print(f"\n{'='*55}")
    print(f"  Final Results ({count} images)")
    print(f"{'='*55}")
    print(f"  MSE        : {shared['mse_sum'] / count:.4f}")
    for t in csi_thresholds:
        TP = shared[f'TP_{t}']
        FP = shared[f'FP_{t}']
        FN = shared[f'FN_{t}']
        den = TP + FP + FN
        csi = TP / den if den > 0 else 1.0
        print(f"  CSI @ {t:3d}  : {csi:.4f}   (TP={TP:,}  FP={FP:,}  FN={FN:,})")
    print(f"{'='*55}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

