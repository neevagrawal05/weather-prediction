"""
ArchesWeather-S + ChaosBench Task 2
====================================
Task:       Autoregressive forecasting of z500 and t850 from day 1 to day 60.
Loss:       Latitude-weighted MSE + masked Spectral Divergence.
Evaluation: RMSE, Bias, ACC, MS-SSIM, Spectral Divergence at selected leads.
Output:     Prediction tensor, metrics CSV, plots, and explicit tensor-shape trace.

NOTE: Defaults to RUN_MODE = 'real' with MODEL_KIND = 'arches'.
      Switch to RUN_MODE = 'demo' for a quick full-pipeline demonstration
      that runs on any machine without large ERA5 data.
"""

# ==============================================================================
# BLOCK 1: Install Python Packages
# ==============================================================================
# Run this once before anything else. Colab may ask for a runtime restart
# if packages were upgraded — restart and re-run from the top.

import sys
import subprocess

packages = [
    'lightning',
    'huggingface_hub',
    'hydra-core',
    'omegaconf',
    'xarray',
    'zarr',
    'pytorch-msssim',
    'timm',
    'axial-attention',
    'tensordict',
    'einops',
    'requests',
    'pandas',
    'matplotlib',
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + packages)
print('Dependencies installed.')


# ==============================================================================
# BLOCK 2: Main Switches
# ==============================================================================
# RUN_MODE = 'demo'  → synthetic but weather-shaped dataset; fast, no large downloads.
# RUN_MODE = 'real'  → official ChaosBench ERA5 zarr files.
# MODEL_KIND = 'compact' → lightweight residual CNN (robust on Colab).
# MODEL_KIND = 'arches'  → frozen ArchesWeather-S backbone + trainable 2-ch head.

RUN_MODE   = 'real'     # 'demo' or 'real'
MODEL_KIND = 'arches'   # 'compact' or 'arches'

# Real ChaosBench settings — keep False until you have enough storage.
DOWNLOAD_CHAOSBENCH              = True
STREAM_EXTRACT_SMALL_REAL_SUBSET = True
REAL_STORES_PER_YEAR             = 365
REAL_CONTIGUOUS_START_DOY        = 1
CLEAN_OLD_CHAOSBENCH_DOWNLOAD    = True
USE_GOOGLE_DRIVE                 = False

# Training controls. Increase epochs for better real-data results.
EPOCHS              = 16  if RUN_MODE == 'demo' else 8
BATCH_SIZE          = 8   if RUN_MODE == 'demo' else 1
LEARNING_RATE       = 3e-4 if RUN_MODE == 'demo' else 5e-4
WEIGHT_DECAY        = 1e-5
SPECTRAL_WEIGHT     = 0.03 if RUN_MODE == 'demo' else 0.01
TRAIN_ROLLOUT_STEPS = 4   if RUN_MODE == 'demo' else 2
MAX_DELTA           = 0.25
STATE_CLAMP         = 3.0  if RUN_MODE == 'demo' else 6.0
ROLLOUT_DAYS        = 60   if RUN_MODE == 'demo' else 14
EVAL_LEADS          = [1, 7, 14, 30, 45, 60] if RUN_MODE == 'demo' else [1, 7, 14]

# Pair / sample limits (remove or increase for paper-quality runs).
MAX_TRAIN_PAIRS              = 800 if RUN_MODE == 'demo' else 140
MAX_VAL_PAIRS                = 80  if RUN_MODE == 'demo' else 50
MAX_STATS_SAMPLES            = 80  if RUN_MODE == 'demo' else 150
MAX_EVAL_INITIAL_CONDITIONS  = 8   if RUN_MODE == 'demo' else 5

TRAIN_YEARS = ['2016', '2017']
VAL_YEARS   = ['2018']

print('RUN_MODE :', RUN_MODE)
print('MODEL_KIND:', MODEL_KIND)


# ==============================================================================
# BLOCK 3: Imports, Paths, Device, Seed
# ==============================================================================

import os
import json
import math
import tarfile
import shutil
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')           # non-interactive backend; swap to 'TkAgg' locally
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pytorch_msssim import ms_ssim, ssim

# ── Google Drive (optional) ────────────────────────────────────────────────────
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/s2s_colab_task2')
else:
    ROOT = Path('/content/s2s_colab_task2')

DATA_DIR = ROOT / 'data' / 'chaosbench'
OUT_DIR  = ROOT / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
if device.type == 'cuda':
    print('GPU        :', torch.cuda.get_device_name(0))
    print('GPU memory:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
print('ROOT:', ROOT)


# ==============================================================================
# BLOCK 4: Shape Tracing and Physics Losses
# ==============================================================================
# shape_trace records the first-seen shape of every named tensor.
# LatWeightedMSE  → downweights polar grid cells (smaller physical area).
# SpectralDivergenceLoss → penalises over-smoothing via radial power spectra.

GRID_H     = 121
GRID_W     = 240
VAR_NAMES  = ['z500', 't850']
shape_trace: dict = {}


def trace(name: str, tensor, note: str = '') -> None:
    """Record tensor metadata the first time we see a given name."""
    if name not in shape_trace:
        shape_trace[name] = {
            'shape' : list(tensor.shape) if hasattr(tensor, 'shape') else None,
            'dtype' : str(tensor.dtype)  if hasattr(tensor, 'dtype')  else type(tensor).__name__,
            'device': str(tensor.device) if hasattr(tensor, 'device') else 'cpu',
            'note'  : note,
        }


def finite_mask(*xs):
    mask = torch.ones_like(xs[0], dtype=torch.bool)
    for x in xs:
        mask = mask & torch.isfinite(x)
    return mask


class LatWeightedMSE(nn.Module):
    """Latitude-weighted mean-squared error."""
    def __init__(self, height: int = GRID_H):
        super().__init__()
        lat = torch.linspace(90.0, -90.0, height)
        w   = torch.cos(torch.deg2rad(lat)).clamp_min(0)
        w   = w / w.mean().clamp_min(1e-8)
        self.register_buffer('w', w.view(1, 1, height, 1), persistent=False)

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = finite_mask(pred, target)
        wf    = self.w.to(pred.device, pred.dtype)
        mf    = mask.to(pred.dtype)
        diff2 = (torch.nan_to_num(pred) - torch.nan_to_num(target)).pow(2)
        return (diff2 * wf * mf).sum() / (wf * mf).sum().clamp_min(1.0)


class SpectralDivergenceLoss(nn.Module):
    """Symmetric KL divergence on normalised radial power spectra."""
    def __init__(self, height: int = GRID_H, width: int = GRID_W,
                 eps: float = 1e-8, min_k_quantile: float = 0.0):
        super().__init__()
        self.height, self.width, self.eps = height, width, eps
        k_lat  = torch.fft.fftfreq(height)  * height
        k_lon  = torch.fft.rfftfreq(width)  * width
        kk     = torch.sqrt(k_lat[:, None] ** 2 + k_lon[None, :] ** 2)
        bins   = kk.round().long()
        valid  = bins > 0
        if min_k_quantile > 0:
            valid = valid & (bins.float() >= torch.quantile(bins[valid].float(), min_k_quantile))
        n_bins = int(bins.max().item()) + 1
        counts = torch.zeros(n_bins)
        counts.scatter_add_(0, bins.reshape(-1), valid.reshape(-1).float())
        self.register_buffer('bins',   bins.reshape(-1),        persistent=False)
        self.register_buffer('valid',  valid.reshape(-1).float(), persistent=False)
        self.register_buffer('counts', counts.clamp_min(1.0),   persistent=False)
        self.register_buffer('keep',   counts > 0,              persistent=False)
        self.n_bins = n_bins

    def _pdf(self, x, mask):
        b, c, h, w = x.shape
        mask_f = mask.to(x.dtype)
        clean  = torch.where(mask, torch.nan_to_num(x), torch.zeros_like(x))
        mean   = clean.sum(dim=(-2, -1), keepdim=True) / mask_f.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        clean  = torch.where(mask, clean - mean, torch.zeros_like(clean))
        power  = torch.fft.rfft2(clean, norm='ortho').abs().pow(2).reshape(b * c, -1)
        power  = power * self.valid.to(power.device, power.dtype).view(1, -1)
        out    = power.new_zeros((b * c, self.n_bins))
        index  = self.bins.to(power.device).view(1, -1).expand(b * c, -1)
        out.scatter_add_(1, index, power)
        out    = out / self.counts.to(power.device, power.dtype).view(1, -1)
        out    = out[:, self.keep.to(power.device)].clamp_min(self.eps)
        return out / out.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = finite_mask(pred, target)
        else:
            mask = torch.broadcast_to(mask.bool() & finite_mask(pred, target), pred.shape)
        pp, tt = self._pdf(pred, mask), self._pdf(target, mask)
        return 0.5 * (
            (tt * (tt.log() - pp.log())).sum(-1) +
            (pp * (pp.log() - tt.log())).sum(-1)
        ).mean()


mse_loss_fn  = LatWeightedMSE().to(device)
sdiv_loss_fn = SpectralDivergenceLoss().to(device)


# ==============================================================================
# BLOCK 5: Demo Dataset and Real ChaosBench Utilities
# ==============================================================================
# DemoS2SDataset  → synthetic two-channel global fields with advective dynamics.
# ChaosBenchZarrTask2 → reads official ChaosBench ERA5 zarr stores.

@dataclass
class NormStats:
    mean:        list
    std:         list
    climatology: np.ndarray


class DemoS2SDataset(Dataset):
    """Synthetic S2S dataset — useful for full-pipeline validation."""
    def __init__(self, n_samples: int = 800, rollout_steps: int = 60, phase0: int = 0):
        self.n_samples     = n_samples
        self.rollout_steps = rollout_steps
        self.phase0        = phase0
        lat = torch.linspace(-math.pi, math.pi, GRID_H).view(GRID_H, 1)
        lon = torch.linspace(0, 2 * math.pi, GRID_W).view(1, GRID_W)
        self.z_base = 0.9 * torch.sin(2 * lat) * torch.cos(3 * lon) + 0.25 * torch.sin(7 * lat + 5 * lon)
        self.t_base = 0.8 * torch.cos(1.5 * lat) * torch.sin(4 * lon) + 0.2 * torch.cos(6 * lat - 3 * lon)

    def _state(self, t: int):
        z        = torch.roll(self.z_base, shifts=int(t),       dims=1)
        q        = torch.roll(self.t_base, shifts=int(0.7 * t), dims=1)
        seasonal = 0.08 * math.sin(2 * math.pi * t / 45.0)
        z        = z + seasonal * torch.roll(self.t_base, shifts=int(t / 2), dims=0)
        q        = q - seasonal * torch.roll(self.z_base, shifts=int(t / 3), dims=0)
        return torch.stack([z, q], dim=0).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = idx + self.phase0
        x = self._state(t)
        y = torch.stack([self._state(t + s + 1) for s in range(self.rollout_steps)], dim=0)
        return {'x': x, 'y': y, 'mask': torch.isfinite(y)}


# ── Streaming multi-URL downloader ────────────────────────────────────────────
class ChainedHTTPStream:
    def __init__(self, urls, chunk_size: int = 8 * 1024 * 1024):
        import requests
        self.urls       = list(urls)
        self.chunk_size = chunk_size
        self.session    = requests.Session()
        self.response   = None
        self.iterator   = None
        self.buffer     = bytearray()
        self.url_idx    = -1

    def _open_next(self):
        self.url_idx += 1
        if self.url_idx >= len(self.urls):
            return False
        url = self.urls[self.url_idx]
        print(f'Streaming ChaosBench chunk {self.url_idx + 1}/{len(self.urls)}: {url.split("/")[-1]}')
        self.response = self.session.get(url, stream=True)
        self.response.raise_for_status()
        self.iterator = self.response.iter_content(chunk_size=self.chunk_size)
        return True

    def read(self, n: int = -1):
        if n is None or n < 0:
            n = self.chunk_size
        while len(self.buffer) < n:
            if self.iterator is None:
                if not self._open_next():
                    break
            try:
                chunk = next(self.iterator)
                if chunk:
                    self.buffer.extend(chunk)
            except StopIteration:
                if self.response is not None:
                    self.response.close()
                self.iterator = None
                self.response = None
        out = bytes(self.buffer[:n])
        del self.buffer[:n]
        return out


def _safe_extract_member(tar, member, path: str) -> None:
    target = (Path(path) / member.name).resolve()
    base   = Path(path).resolve()
    if not str(target).startswith(str(base)):
        raise RuntimeError(f'Blocked unsafe tar path: {member.name}')
    tar.extract(member, path=path)


def _date_from_zarr_root(root) -> 'pd.Timestamp | None':
    m = re.search(r'(20\d{6})', Path(root).name)
    return None if m is None else pd.to_datetime(m.group(1), format='%Y%m%d')


def _target_dates_for_years(years, start_doy: int = 1, days_per_year: int = 45) -> dict:
    targets = {}
    for year in years:
        start = pd.Timestamp(int(year), 1, 1) + pd.Timedelta(days=int(start_doy) - 1)
        targets[str(year)] = set(
            (start + pd.Timedelta(days=i)).strftime('%Y%m%d') for i in range(int(days_per_year))
        )
    return targets


def _zarr_root_and_year(member_name: str, years):
    parts      = member_name.replace('\\', '/').split('/')
    root_parts = []
    root       = None
    for part in parts:
        root_parts.append(part)
        if part.endswith('.zarr'):
            root = '/'.join(root_parts)
            break
    if root is None:
        return None, None
    for year in years:
        if str(year) in root:
            return root, str(year)
    return root, None


def _normalize_extracted_era5_dir(data_dir) -> Path:
    data_dir = Path(data_dir)
    era5_dir = data_dir / 'era5'
    if era5_dir.exists() and any(era5_dir.glob('*.zarr')):
        return era5_dir
    for candidate in [data_dir / 'era5_tmp', data_dir / 'era5']:
        if candidate.exists() and candidate != era5_dir:
            if era5_dir.exists():
                for item in candidate.glob('*.zarr'):
                    shutil.move(str(item), str(era5_dir / item.name))
            else:
                candidate.rename(era5_dir)
    era5_dir.mkdir(parents=True, exist_ok=True)
    return era5_dir


def cleanup_old_full_download(data_dir) -> None:
    data_dir = Path(data_dir)
    for pattern in ['era5.tar.gz', 'era5/era5_chunks.tar.gz.*']:
        for p in data_dir.glob(pattern):
            if p.is_file():
                p.unlink()
                print('Deleted old full-download file:', p)


def download_chaosbench_era5(data_dir, years, max_stores_per_year: int = 45,
                              start_doy: int = 1, stream_small_subset: bool = True,
                              clean_old: bool = True):
    from huggingface_hub import hf_hub_url
    years    = [str(y) for y in years]
    data_dir = Path(data_dir)
    era5_dir = data_dir / 'era5'
    data_dir.mkdir(parents=True, exist_ok=True)
    if clean_old:
        cleanup_old_full_download(data_dir)

    target_dates = _target_dates_for_years(years, start_doy=start_doy, days_per_year=max_stores_per_year)
    existing_ok  = True
    for y in years:
        existing_dates = set()
        for p in era5_dir.glob(f'*{y}*.zarr'):
            d = _date_from_zarr_root(p.name)
            if d is not None:
                existing_dates.add(d.strftime('%Y%m%d'))
        existing_ok = existing_ok and target_dates[y].issubset(existing_dates)
    if existing_ok:
        print('Small real subset already exists:', era5_dir)
        return data_dir

    if not stream_small_subset:
        raise RuntimeError(
            'Full ChaosBench download is disabled to avoid Colab disk overflow. '
            'Use stream_small_subset=True.'
        )

    urls = [
        hf_hub_url(repo_id='LEAP/ChaosBench', repo_type='dataset',
                   filename=f'era5/era5_chunks.tar.gz.{s}')
        for s in ['aa', 'ab']
    ]
    selected       = {y: [] for y in years}
    selected_roots = set()
    stream         = ChainedHTTPStream(urls)

    print(f'Extracting contiguous windows: {max_stores_per_year} daily zarr stores '
          f'per year from day-of-year {start_doy}.')
    print('Streams remote tarballs and writes only selected date-window zarr stores to disk.')

    with tarfile.open(fileobj=stream, mode='r|gz') as tar:
        done_after_current_root = False
        for member in tar:
            root, year = _zarr_root_and_year(member.name, years)
            if done_after_current_root and (root not in selected_roots):
                print('Reached requested subset size; stopping stream extraction.')
                break
            should_extract = False
            if root is not None and year is not None:
                root_date = _date_from_zarr_root(root)
                date_key  = None if root_date is None else root_date.strftime('%Y%m%d')
                if root in selected_roots:
                    should_extract = True
                elif date_key in target_dates[year]:
                    selected_roots.add(root)
                    selected[year].append(root)
                    should_extract = True
                    print(f'  selected {year}: {len(selected[year])}/{max_stores_per_year} '
                          f'-> {Path(root).name}')
            if should_extract:
                _safe_extract_member(tar, member, data_dir)
            if all(len(v) >= max_stores_per_year for v in selected.values()):
                done_after_current_root = True

    era5_dir = _normalize_extracted_era5_dir(data_dir)
    counts   = {y: len(list(era5_dir.glob(f'*{y}*.zarr'))) for y in years}
    print('Extracted zarr counts:', counts)

    for y in years:
        got_dates = set()
        for p in era5_dir.glob(f'*{y}*.zarr'):
            d = _date_from_zarr_root(p.name)
            if d is not None:
                got_dates.add(d.strftime('%Y%m%d'))
        missing = sorted(target_dates[y] - got_dates)
        if missing:
            raise RuntimeError(
                f'Missing {len(missing)} requested contiguous dates for {y}; '
                f'first missing: {missing[:5]}'
            )
    return data_dir


def find_year_files(data_dir, years) -> list:
    era5  = Path(data_dir) / 'era5'
    files = []
    for y in years:
        files += list(era5.glob(f'*{y}*.zarr'))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f'No zarr files for {years} under {era5}')
    return files


def _select_level(da, level):
    for dim in ['level', 'pressure_level', 'isobaricInhPa', 'plev']:
        if dim in da.dims:
            values = np.asarray(da.coords[dim].values).astype(float)
            return da.isel({dim: int(np.abs(values - level).argmin())})
    return da


def extract_headline_fields(path) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(path, engine='zarr', chunks=None)
    try:
        z = ds['z-500'] if 'z-500' in ds else _select_level(ds['z'], 500)
        t = ds['t-850'] if 't-850' in ds else _select_level(ds['t'], 850)
        return np.stack([
            np.asarray(z).squeeze().reshape(GRID_H, GRID_W),
            np.asarray(t).squeeze().reshape(GRID_H, GRID_W),
        ], axis=0).astype('float32')
    finally:
        ds.close()


def compute_stats(files, max_samples: int = 80) -> NormStats:
    total    = np.zeros(2)
    total2   = np.zeros(2)
    count    = np.zeros(2)
    clim_sum = np.zeros((2, GRID_H, GRID_W))
    clim_cnt = np.zeros((2, GRID_H, GRID_W))
    for path in tqdm(files[:max_samples], desc='Computing normalisation stats'):
        arr   = extract_headline_fields(path).astype('float64')
        valid = np.isfinite(arr)
        safe  = np.where(valid, arr, 0.0)
        total  += safe.reshape(2, -1).sum(1)
        total2 += (safe * safe).reshape(2, -1).sum(1)
        count  += valid.reshape(2, -1).sum(1)
        clim_sum += safe
        clim_cnt += valid
    mean = total / count
    std  = np.sqrt(np.maximum(total2 / count - mean * mean, 1e-12))
    clim = (clim_sum / np.maximum(clim_cnt, 1.0) - mean[:, None, None]) / std[:, None, None]
    print(f'Stats — mean: {mean.tolist()}, std: {std.tolist()}')
    return NormStats(mean.tolist(), std.tolist(), clim.astype('float32'))


class ChaosBenchZarrTask2(Dataset):
    """Official ChaosBench ERA5 dataset for Task-2 autoregressive rollout."""
    def __init__(self, files, stats: NormStats, rollout_steps: int = 60,
                 max_pairs=None):
        self.files         = sorted(list(files))
        self.stats         = stats
        self.rollout_steps = rollout_steps

        date_to_file = {}
        for f in self.files:
            d = _date_from_zarr_root(str(f))
            if d is not None:
                date_to_file[d.strftime('%Y%m%d')] = f

        starts = []
        for key, f in sorted(date_to_file.items()):
            d0     = pd.to_datetime(key, format='%Y%m%d')
            needed = [(d0 + pd.Timedelta(days=s)).strftime('%Y%m%d')
                      for s in range(rollout_steps + 1)]
            if all(k in date_to_file for k in needed):
                starts.append([date_to_file[k] for k in needed])

        self.sequences = starts[:max_pairs] if max_pairs is not None else starts
        if not self.sequences:
            raise RuntimeError(
                'No contiguous daily sequences found. '
                'Increase REAL_STORES_PER_YEAR or clean/re-extract data.'
            )

    def __len__(self):
        return len(self.sequences)

    def norm(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            ((arr - np.array(self.stats.mean)[:, None, None]) /
             (np.array(self.stats.std)[:, None, None] + 1e-8)).astype('float32')
        )

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x   = self.norm(extract_headline_fields(seq[0]))
        y   = torch.stack([self.norm(extract_headline_fields(seq[s + 1]))
                           for s in range(self.rollout_steps)], dim=0)
        return {'x': x, 'y': y, 'mask': torch.isfinite(y)}


# ==============================================================================
# BLOCK 6: Compact Residual Model
# ==============================================================================
# Residual CNN: predicts a one-day correction  x(t) → x(t+1).
# Output shape: [B, 2, 121, 240]  (same as input).

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class CompactTask2Net(nn.Module):
    def __init__(self, width: int = 96, depth: int = 6):
        super().__init__()
        self.stem   = nn.Conv2d(2, width, 5, padding=2, padding_mode='circular')
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.head   = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1, padding_mode='circular'),
        )

    def forward(self, x):
        trace('forward.input', x, '[B, 2, 121, 240] normalized z500/t850')
        h     = self.stem(x)
        trace('forward.tokens_or_features', h, '[B, width, 121, 240] compact feature map')
        delta = self.head(self.blocks(h))
        if MAX_DELTA is not None:
            delta = MAX_DELTA * torch.tanh(delta / MAX_DELTA)
        out = x + delta
        if STATE_CLAMP is not None:
            out = out.clamp(-STATE_CLAMP, STATE_CLAMP)
        trace('forward.output', out, '[B, 2, 121, 240] one-day prediction')
        return out


def count_params(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {'trainable': trainable, 'frozen': frozen, 'total': trainable + frozen}


# ==============================================================================
# BLOCK 7: Optional ArchesWeather-S Loader
# ==============================================================================
# Downloads config+weights, freezes backbone, hooks internal features,
# trains only a small two-channel task head.

def build_archesweather_s_wrapper():
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf, open_dict
    from hydra.utils import instantiate

    if not Path('/content/ArchesWeather').exists():
        subprocess.check_call([
            'git', 'clone', '--depth', '1',
            'https://github.com/gcouairon/ArchesWeather.git',
            '/content/ArchesWeather',
        ])
    sys.path.insert(0, '/content/ArchesWeather')

    repo = 'gcouairon/ArchesWeather'
    rev  = 'd0718cb8528a61f19d62ade9bf0eab6cff79078d'
    model_dir = str(ROOT / 'modelstore' / 'archesweather-S')
    cfg_path = hf_hub_download(repo_id=repo, filename='archesweather-S_config.yaml',
                                revision=rev, local_dir=model_dir)
    w_path   = hf_hub_download(repo_id=repo, filename='archesweather-S_weights.pt',
                                revision=rev, local_dir=model_dir)

    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.module.backbone.pop('mlp_vert', None)
    backbone = instantiate(cfg.module.backbone)

    state = torch.load(w_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = {k.replace('backbone.', '', 1): v
                 for k, v in state['state_dict'].items()
                 if k.startswith('backbone.')}
    backbone.load_state_dict(state, strict=False)

    emb_dim    = int(cfg.module.backbone.get('emb_dim',   192))
    cond_dim   = int(cfg.module.backbone.get('cond_dim',  256))
    surface_ch = int(cfg.module.backbone.get('surface_ch', 11))
    level_ch   = int(cfg.module.backbone.get('level_ch',   12))

    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    class ArchesWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.emb_dim  = emb_dim
            self.captured = None

            target = None
            for _, m in reversed(list(self.backbone.named_modules())):
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    inc  = getattr(m, 'in_features',  getattr(m, 'in_channels',  None))
                    outc = getattr(m, 'out_features', getattr(m, 'out_channels', None))
                    if inc == emb_dim and outc is not None and outc < 200:
                        target = m
                        break
            if target is None:
                raise RuntimeError('Could not locate ArchesWeather projection head.')
            target.register_forward_pre_hook(
                lambda _m, inp: setattr(self, 'captured', inp[0])
            )

            self.head = nn.Sequential(
                nn.Conv2d(emb_dim,      emb_dim // 2, 3, padding=1), nn.GELU(),
                nn.Conv2d(emb_dim // 2, emb_dim // 4, 3, padding=1), nn.GELU(),
                nn.Conv2d(emb_dim // 4, 2,             1),
            )

        def forward(self, x):
            b    = x.shape[0]
            xa   = F.interpolate(x, size=(120, 240), mode='bilinear', align_corners=False)
            surf = torch.zeros(b, surface_ch, 120, 240, device=x.device, dtype=x.dtype)
            lev  = torch.zeros(b, level_ch, 13, 120, 240, device=x.device, dtype=x.dtype)
            lev[:, 0, 7]  = xa[:, 0]   # z500
            lev[:, 3, 10] = xa[:, 1]   # t850
            cond = torch.zeros(b, cond_dim, device=x.device, dtype=x.dtype)

            trace('arches.input_surface', surf, '[B, surface_ch, 120, 240]')
            trace('arches.input_level',   lev,  '[B, level_ch, 13, 120, 240]')

            with torch.no_grad():
                _ = self.backbone(surf, lev, cond)

            feat = self.captured
            if feat.dim() == 3:
                groups = feat.shape[1] // (120 * 240)
                feat   = feat.view(b, groups, 120, 240, emb_dim)[:, 0].permute(0, 3, 1, 2)
            elif feat.dim() == 4 and feat.shape[-1] == emb_dim:
                feat = feat.permute(0, 3, 1, 2)
            feat = F.interpolate(feat, size=(121, 240), mode='bilinear', align_corners=False)
            trace('arches.features', feat, '[B, embed_dim, 121, 240]')

            delta = self.head(feat)
            if MAX_DELTA is not None:
                delta = MAX_DELTA * torch.tanh(delta / MAX_DELTA)
            out = x + delta
            if STATE_CLAMP is not None:
                out = out.clamp(-STATE_CLAMP, STATE_CLAMP)
            trace('forward.output', out, '[B, 2, 121, 240] residual ArchesWeather prediction')
            return out

    model = ArchesWrapper()
    model.meta = {
        'embed_dim': emb_dim, 'surface_ch': surface_ch,
        'level_ch':  level_ch, 'cond_dim':   cond_dim,
        'patch_size': cfg.module.backbone.get('patch_size', None),
    }
    return model


# ==============================================================================
# BLOCK 8: Build Data Loaders
# ==============================================================================

print('\n--- Building data loaders ---')

if RUN_MODE == 'demo':
    stats    = NormStats([0.0, 0.0], [1.0, 1.0],
                         np.zeros((2, GRID_H, GRID_W), dtype='float32'))
    train_ds = DemoS2SDataset(MAX_TRAIN_PAIRS, rollout_steps=TRAIN_ROLLOUT_STEPS, phase0=0)
    val_ds   = DemoS2SDataset(MAX_VAL_PAIRS,   rollout_steps=ROLLOUT_DAYS,        phase0=2000)
else:
    if DOWNLOAD_CHAOSBENCH:
        download_chaosbench_era5(
            DATA_DIR,
            years=sorted(set(TRAIN_YEARS + VAL_YEARS)),
            max_stores_per_year=REAL_STORES_PER_YEAR,
            start_doy=REAL_CONTIGUOUS_START_DOY,
            stream_small_subset=STREAM_EXTRACT_SMALL_REAL_SUBSET,
            clean_old=CLEAN_OLD_CHAOSBENCH_DOWNLOAD,
        )
    train_files = find_year_files(DATA_DIR, TRAIN_YEARS)
    val_files   = find_year_files(DATA_DIR, VAL_YEARS)

    stats_file = OUT_DIR / (
        f'normalization_task2_window{REAL_STORES_PER_YEAR}'
        f'_doy{REAL_CONTIGUOUS_START_DOY}.npz'
    )
    if stats_file.exists():
        z      = np.load(stats_file)
        stats  = NormStats(z['mean'].tolist(), z['std'].tolist(), z['climatology'])
        print('Loaded cached normalisation stats:', stats_file)
    else:
        stats = compute_stats(train_files, MAX_STATS_SAMPLES)
        np.savez_compressed(
            stats_file,
            mean=np.array(stats.mean),
            std=np.array(stats.std),
            climatology=stats.climatology,
        )
        print('Saved normalisation stats:', stats_file)

    train_ds = ChaosBenchZarrTask2(
        train_files, stats, rollout_steps=TRAIN_ROLLOUT_STEPS, max_pairs=MAX_TRAIN_PAIRS
    )
    val_ds = ChaosBenchZarrTask2(
        val_files, stats, rollout_steps=ROLLOUT_DAYS, max_pairs=MAX_VAL_PAIRS
    )

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=0)

sample = next(iter(train_loader))
trace('data.batch.x', sample['x'], '[B, 2, 121, 240]')
trace('data.batch.y', sample['y'], '[B, T, 2, 121, 240]')
print('Train pairs :', len(train_ds), '| Val pairs:', len(val_ds))
print('x shape      :', tuple(sample['x'].shape))
print('y shape      :', tuple(sample['y'].shape))


# ==============================================================================
# BLOCK 9: Build Model
# ==============================================================================

print('\n--- Building model ---')

if MODEL_KIND == 'arches':
    try:
        model = build_archesweather_s_wrapper()
        print('Loaded ArchesWeather-S frozen backbone.')
    except Exception as exc:
        print('ArchesWeather-S load failed, falling back to compact model:', repr(exc))
        model = CompactTask2Net(width=96, depth=6)
else:
    model = CompactTask2Net(width=96, depth=6)

model = model.to(device)
param_info = count_params(model)
print(json.dumps(param_info, indent=2))
if hasattr(model, 'meta'):
    print('Model meta:', model.meta)


# ==============================================================================
# BLOCK 10: Training Loop
# ==============================================================================
# Loss: latitude_weighted_mse + SPECTRAL_WEIGHT * spectral_divergence
# One-step autoregressive training with gradient clipping and mixed precision.

print('\n--- Training ---')

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))
scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
history   = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(train_loader, desc=f'epoch {epoch}/{EPOCHS}')
    for batch in pbar:
        x    = batch['x'].to(device).float()
        y    = batch['y'].to(device).float()
        mask = batch['mask'].to(device).bool()
        curr = x
        loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            for step in range(TRAIN_ROLLOUT_STEPS):
                pred   = model(curr)
                target = y[:, step]
                m      = mask[:, step]
                l_mse  = mse_loss_fn(pred, target, m)
                l_sdiv = sdiv_loss_fn(pred, target, m)
                loss   = loss + l_mse + SPECTRAL_WEIGHT * l_sdiv
                curr   = pred
            loss = loss / TRAIN_ROLLOUT_STEPS
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        n     += 1
        pbar.set_postfix(loss=f'{total / max(n, 1):.6f}')
    scheduler.step()
    avg = total / max(n, 1)
    history.append({'epoch': epoch, 'train_loss': avg, 'lr': scheduler.get_last_lr()[0]})
    print(f'Epoch {epoch:>2}: train_loss={avg:.6f}, lr={scheduler.get_last_lr()[0]:.2e}')

pd.DataFrame(history).to_csv(OUT_DIR / 'training_history.csv', index=False)
torch.save(
    {'model_state': model.state_dict(), 'params': param_info, 'stats': stats.__dict__},
    OUT_DIR / 'task2_model.pt',
)

# Training loss curve
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], marker='o')
ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.grid(True)
ax.set_title('Training Loss')
fig.tight_layout()
fig.savefig(OUT_DIR / 'training_loss.png', dpi=120)
plt.close(fig)
print('Saved training_loss.png')

# Print training history table
print('\nTraining history:')
print(pd.DataFrame(history).to_string(index=False))


# ==============================================================================
# BLOCK 11: Autoregressive Evaluation
# ==============================================================================
# Prediction tensor shape: [N_initial_conditions, rollout_days, 2, 121, 240]
# Metrics: RMSE, Bias, ACC, MS-SSIM, Spectral Divergence at selected lead days.

print('\n--- Autoregressive evaluation ---')


def compute_msssim_np(p: np.ndarray, y: np.ndarray) -> float:
    lo, hi = -3.0, 3.0
    pt = torch.tensor(np.clip((p - lo) / (hi - lo), 0.0, 1.0)).float()[None, None]
    yt = torch.tensor(np.clip((y - lo) / (hi - lo), 0.0, 1.0)).float()[None, None]
    try:
        return float(ms_ssim(pt, yt, data_range=1.0))
    except Exception:
        return float(ssim(pt, yt, data_range=1.0))


def metric_row(pred: np.ndarray, target: np.ndarray,
               clim: np.ndarray, lead: int, var: str) -> dict:
    p  = pred.astype('float64')
    y  = target.astype('float64')
    c  = clim.astype('float64')
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    bias = float(np.mean(p - y))
    pa, ya = p - c, y - c
    acc  = float(np.sum(pa * ya) /
                 (np.sqrt(np.sum(pa ** 2) * np.sum(ya ** 2)) + 1e-12))
    with torch.no_grad():
        sdiv = float(
            sdiv_loss_fn(
                torch.tensor(p)[None, None].float().to(device),
                torch.tensor(y)[None, None].float().to(device),
            ).cpu()
        )
    return {
        'lead_day':    lead,
        'variable':    var,
        'rmse_norm':   rmse,
        'bias_norm':   bias,
        'acc':         acc,
        'ms_ssim':     compute_msssim_np(p, y),
        'spectral_div': sdiv,
    }


@torch.no_grad()
def rollout_one(x0: torch.Tensor, days: int) -> torch.Tensor:
    model.eval()
    curr = x0[None].to(device).float()
    outs = []
    for _ in range(days):
        curr = model(curr)
        outs.append(curr.squeeze(0).detach().cpu())
    return torch.stack(outs, dim=0)


predictions, targets, rows = [], [], []
n_eval = min(MAX_EVAL_INITIAL_CONDITIONS, len(val_ds))
clim   = stats.climatology

for i in tqdm(range(n_eval), desc='rollout'):
    item = val_ds[i]
    pred = rollout_one(item['x'], ROLLOUT_DAYS)
    targ = item['y'][:ROLLOUT_DAYS]
    trace('eval.prediction_tensor_one_ic', pred, '[rollout_days, 2, 121, 240]')
    predictions.append(pred.numpy())
    targets.append(targ.numpy())
    for lead in EVAL_LEADS:
        if lead <= pred.shape[0]:
            for ch, name in enumerate(VAR_NAMES):
                row = {'initial_condition': i}
                row.update(metric_row(
                    pred[lead - 1, ch].numpy(),
                    targ[lead - 1, ch].numpy(),
                    clim[ch], lead, name,
                ))
                rows.append(row)

prediction_np = np.stack(predictions, axis=0)
target_np     = np.stack(targets,     axis=0)
np.savez_compressed(
    OUT_DIR / 'predictions_task2_z500_t850.npz',
    prediction_norm=prediction_np,
    target_norm=target_np,
    lead_days=np.arange(1, ROLLOUT_DAYS + 1),
    variable_names=np.array(VAR_NAMES),
)
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT_DIR / 'metrics_task2_z500_t850.csv', index=False)

print('\nPrediction tensor shape:', prediction_np.shape)
print('\nMetrics summary (mean over initial conditions):')
summary = (
    metrics
    .groupby(['lead_day', 'variable'])[['rmse_norm', 'bias_norm', 'acc', 'ms_ssim', 'spectral_div']]
    .mean()
    .reset_index()
)
print(summary.to_string(index=False))


# ==============================================================================
# BLOCK 12: Plots and Saved Outputs
# ==============================================================================

print('\n--- Generating plots ---')

# ── Metric curves ─────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 4, figsize=(16, 3))
col_labels = [
    ('rmse_norm',    'RMSE (lower better)'),
    ('acc',          'ACC (higher better)'),
    ('ms_ssim',      'MS-SSIM (higher better)'),
    ('spectral_div', 'SpecDiv (lower better)'),
]
for ax, (col, title) in zip(axs, col_labels):
    for var in VAR_NAMES:
        s = summary[summary.variable == var]
        ax.plot(s.lead_day, s[col], marker='o', label=var)
    ax.set_title(title)
    ax.set_xlabel('lead day')
    ax.grid(True)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'metric_curves.png', dpi=120)
plt.close(fig)
print('Saved metric_curves.png')

# ── Prediction / target / error maps ─────────────────────────────────────────
lead_to_plot = min(14, ROLLOUT_DAYS)
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for ch, var in enumerate(VAR_NAMES):
    p = prediction_np[0, lead_to_plot - 1, ch]
    y = target_np    [0, lead_to_plot - 1, ch]
    e = p - y
    for ax, arr, title in zip(
        axs[ch],
        [p,                           y,                              e],
        [f'{var} pred day {lead_to_plot}',
         f'{var} target day {lead_to_plot}',
         f'{var} error'],
    ):
        im = ax.imshow(arr, cmap='RdBu_r')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(OUT_DIR / 'prediction_maps.png', dpi=120)
plt.close(fig)
print('Saved prediction_maps.png')

# ── Shape trace & model parameter summary ────────────────────────────────────
with open(OUT_DIR / 'shape_trace.json', 'w') as f:
    json.dump(shape_trace, f, indent=2)
with open(OUT_DIR / 'model_parameters.json', 'w') as f:
    json.dump(
        {
            'params':      param_info,
            'run_mode':    RUN_MODE,
            'model_kind':  MODEL_KIND,
            'rollout_days': ROLLOUT_DAYS,
            'eval_leads':  EVAL_LEADS,
        },
        f, indent=2,
    )

print('\nShape trace:')
print(json.dumps(shape_trace, indent=2))

print('\nModel parameters:')
print(json.dumps(param_info, indent=2))

print(f'\nAll outputs saved to: {OUT_DIR}')
print('  metrics_task2_z500_t850.csv')
print('  predictions_task2_z500_t850.npz')
print('  training_history.csv')
print('  training_loss.png')
print('  metric_curves.png')
print('  prediction_maps.png')
print('  shape_trace.json')
print('  model_parameters.json')
print('  task2_model.pt')


# ==============================================================================
# BLOCK 13: Instructions — Moving from Demo to Real ChaosBench
# ==============================================================================
"""
To switch from demo to the official benchmark:

1.  Set  RUN_MODE = 'real'
2.  Set  USE_GOOGLE_DRIVE = True  (if Colab local disk is too small)
3.  Set  DOWNLOAD_CHAOSBENCH = True  (first run only — streams ERA5 tarballs)
4.  Keep MODEL_KIND = 'compact' to verify data loading first.
5.  Then try MODEL_KIND = 'arches' for frozen ArchesWeather-S features.
6.  Increase EPOCHS, MAX_TRAIN_PAIRS, MAX_EVAL_INITIAL_CONDITIONS
    for report-quality results.

Demo-mode numbers are pipeline-validation only.
Report ONLY real-mode metrics as ChaosBench benchmark scores.
"""
