import os
import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import eval_metrics as utpp
from eval_metrics import csi, far, hss, rmse, crps, fss

GT_DIR = "./bbdm_output/gt"
PRED_DIR = "./bbdm_output/pred"
LOG_OUT = "./bbdm_output/eval.log"
T_OUT    = 12    # frames per sequence


class MetricListEvaluator():
    def __init__(self, metric_list):
        self.metric_holder = {}
        self.batch_count = 0
        for metric_name in metric_list:
            threshold = ''
            if '-' in metric_name:
                parts = metric_name.split('-')
                base_name = parts[0]
                threshold = parts[1]
            else:
                base_name = metric_name

            key_name = metric_name
            thresh_val = float(threshold) / 255.0 if threshold.isdigit() else None
            self.metric_holder[key_name] = self.init_metric(base_name, threshold=thresh_val)

    def init_metric(self, metric_name, **kwarg):
        if metric_name in ['csi', 'pod', 'far', 'hss']:
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float64), {'threshold': kwarg['threshold']}]
        elif metric_name == 'csi_4':
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float64), {'threshold': kwarg['threshold'], 'radius': 4}]
        elif metric_name == 'csi_16':
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float64), {'threshold': kwarg['threshold'], 'radius': 16}]
        else:
            func = globals().get(metric_name, getattr(utpp, metric_name, None))
            if func is None:
                raise ValueError(f"Metric function {metric_name} not found.")
            return [func, 0.0, {}]

    def eval(self, y_pred, y):
        self.batch_count += 1
        for key, metric in self.metric_holder.items():
            func, val_acc, kwargs = metric
            temp = func(y_pred, y, **kwargs)
            if isinstance(temp, (list, tuple, np.ndarray)):
                metric[1] += np.array(temp)
            elif isinstance(temp, torch.Tensor):
                metric[1] += temp.detach().cpu().item()
            else:
                metric[1] += temp

    def get_results(self):
        output_holder = {}
        for key, metric in self.metric_holder.items():
            func, val_acc, kwargs = metric
            base_name = key.split('-')[0]
            if func is utpp.tfpn:
                calc_func_name = base_name.split('_')[0]
                calc_func = getattr(utpp, calc_func_name)
                final_score = calc_func(*list(val_acc))
                output_holder[key] = final_score
            else:
                output_holder[key] = val_acc / self.batch_count if self.batch_count > 0 else 0.0
        return output_holder


def load_png_frames(directory: str) -> np.ndarray:
    """Load all PNGs sorted → float32 (N, H, W) normalized to 0-1."""
    files = sorted(Path(directory).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png files in {directory}")
    return np.stack(
        [np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
         for f in tqdm(files, desc=f"Loading {directory}", leave=False)],
        axis=0
    )  # (N, H, W)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir',   type=str, default=GT_DIR)
    parser.add_argument('--pred_dir', type=str, default=PRED_DIR)
    parser.add_argument('--t_out',    type=int, default=T_OUT)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output',   type=str, default=LOG_OUT)
    args = parser.parse_args()

    # 1. Logging
    logfile = args.output
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"GT:   {args.gt_dir}")
    logging.info(f"Pred: {args.pred_dir}")

    # 2. Load PNGs  →  (N, H, W) in 0-1
    logging.info("Loading PNG frames...")
    gt_frames   = load_png_frames(args.gt_dir)
    pred_frames = load_png_frames(args.pred_dir)

    N = len(gt_frames)
    assert len(pred_frames) == N, f"Frame count mismatch: gt={N}, pred={len(pred_frames)}"
    logging.info(f"  {N} frames | shape {gt_frames.shape[1:]}")

    n_seq = N // args.t_out
    if n_seq == 0:
        raise RuntimeError(f"Not enough frames ({N}) for T_OUT={args.t_out}")
    if N % args.t_out:
        logging.warning(f"Last {N % args.t_out} frames ignored")

    # Reshape → (n_seq, T_OUT, 1, H, W)  — add channel dim to match (B, T, C, H, W)
    H, W = gt_frames.shape[1], gt_frames.shape[2]
    gt_seq   = gt_frames  [:n_seq * args.t_out].reshape(n_seq, args.t_out, 1, H, W)
    pred_seq = pred_frames[:n_seq * args.t_out].reshape(n_seq, args.t_out, 1, H, W)
    logging.info(f"  {n_seq} sequences × {args.t_out} frames/seq\n")

    # 3. Metrics
    metrics_list = [
        'csi-16', 'csi-74', 'csi-133', 'csi-160', 'csi-181', 'csi-219',
        'csi_4-16', 'csi_4-74', 'csi_4-133', 'csi_4-160', 'csi_4-181', 'csi_4-219',
        'csi_16-16', 'csi_16-74', 'csi_16-133', 'csi_16-160', 'csi_16-181', 'csi_16-219',
        'hss-16', 'hss-74', 'hss-133', 'hss-160', 'hss-181', 'hss-219',
        'far-16', 'far-74', 'far-133', 'far-160', 'far-181', 'far-219',
        'rmse', 'crps', 'fss'
    ]
    evaluator = MetricListEvaluator(metrics_list)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 4. Evaluation loop (batched over sequences)
    for i in tqdm(range(0, n_seq, args.batch_size), desc="Evaluating"):
        y_gt   = torch.from_numpy(gt_seq  [i:i + args.batch_size]).to(device)
        y_pred = torch.from_numpy(pred_seq[i:i + args.batch_size]).to(device)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)
        evaluator.eval(y_pred, y_gt)

    # 5. Results
    logging.info("=========================================")
    logging.info("       Final Evaluation Results          ")
    logging.info("=========================================")
    final_results = evaluator.get_results()
    for k, v in final_results.items():
        logging.info(f'{k}: {v:.6f}')
    logging.info(f"Log saved to {logfile}")

