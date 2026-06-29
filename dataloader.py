import sys
import os
import numpy as np
import datetime

# Compatibility fix for NumPy 1.20+ (removes np.float alias)
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_

from sevir.sevir_torch_wrap import SEVIRTorchDataset

catalog_path = './sevir/CATALOG.csv'
data_dir = './sevir/data'

# SEVIR-LR files store 25 frames at 10-minute intervals.
# We use exactly 12 contiguous frames: 6 for input + 6 for target.
seq_len = 25
stride = 12
sample_mode = 'sequent'
layout = 'THWC'

start_date = datetime.datetime(2019, 6, 1)


def build_sevir():
    ds_train = SEVIRTorchDataset(
        sevir_catalog=catalog_path,
        sevir_data_dir=data_dir,
        raw_seq_len=49,
        split_mode='uneven',
        shuffle=True,
        seq_len=seq_len,
        stride=stride,
        sample_mode=sample_mode,
        layout=layout,
        start_date=None,
        end_date=start_date,
        output_type=np.float32,
        preprocess=True,
        rescale_method='01',
        verbose=False,
        aug_mode='0',
        ret_contiguous=True,
    )

    ds_test = SEVIRTorchDataset(
        sevir_catalog=catalog_path,
        sevir_data_dir=data_dir,
        raw_seq_len=49,
        split_mode='uneven',
        shuffle=False,
        seq_len=seq_len,
        stride=stride,
        sample_mode=sample_mode,
        layout=layout,
        start_date=start_date,
        end_date=None,
        output_type=np.float32,
        preprocess=True,
        rescale_method='01',
        verbose=False,
        aug_mode='0',
        ret_contiguous=True,
    )

    return ds_train, ds_test

if __name__ == '__main__':
    train_ds, test_ds = build_sevir()
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    sample = test_ds[219]
    print(sample.shape)
