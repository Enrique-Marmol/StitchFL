#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Membership inference attack benchmark for multiclass classification.

Implements Yeom, LiRA, RMIA, Attack-R, Quantile, and TrajectoryMIA with stratified membership sampling, shared shadow models, bootstrap AUC confidence intervals, and low-FPR evaluation."""
import os
import sys
import csv
import glob
import json
import time
import math
import hashlib
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from scipy.stats import norm, beta
from scipy.special import expit
ATTACKS = ['Yeom', 'LiRA', 'RMIA', 'Attack-R', 'Quantile', 'TMIA']

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'Datasets' / 'letter+recognition'
DEFAULT_DATA_FILE = DATA_DIR / 'letter-recognition.data'
DEFAULT_OUTPUT_DIR = BASE_DIR / 'results' / 'mia_classification'

def load_dataset(data_root: str, csv_path: str, label_col: int, has_header: bool):
    if csv_path:
        paths = [csv_path]
    else:
        paths = sorted(glob.glob(os.path.join(data_root, '*.data')))
        if not paths:
            paths = sorted(glob.glob(os.path.join(data_root, 'client_*.csv')))
        if not paths:
            paths = sorted(glob.glob(os.path.join(data_root, '*.csv')))
        if not paths:
            raise FileNotFoundError(f'No .data or .csv files found in {data_root}')
    header = 0 if has_header else None
    df = pd.concat([pd.read_csv(p, header=header) for p in paths], ignore_index=True)
    print(f'[data] {len(paths)} file(s), {len(df)} rows, {df.shape[1]} columns')
    if df.shape[1] < 2:
        raise ValueError('The dataset must contain at least one label column and one feature column.')
    lab_name = df.columns[label_col]
    raw = df[lab_name].astype(str).str.strip()
    feats = df.drop(columns=[lab_name])
    X = feats.apply(pd.to_numeric, errors='coerce').to_numpy(dtype='float64')
    ok = (raw != '') & (raw.str.lower() != 'nan')
    X, raw = (X[ok.to_numpy()], raw[ok.to_numpy()])
    if (~ok).any():
        print(f'[data] dropped {int((~ok).sum())} rows without labels')
    classes = np.array(sorted(pd.unique(raw)))
    y = np.searchsorted(classes, raw.to_numpy()).astype('int64')
    bad = ~np.isfinite(X)
    if bad.any():
        mu = np.nanmean(np.where(bad, np.nan, X), axis=0)
        mu[~np.isfinite(mu)] = 0.0
        X[bad] = np.take(mu, np.where(bad)[1])
        print(f'[data] imputed {int(bad.sum())} non-finite feature values')
    cnt = np.bincount(y, minlength=len(classes))
    print(f'[data] final: X={X.shape}  classes={len(classes)}  ({classes[0]}..{classes[-1]})')
    print(f'[data] samples per class: min={cnt.min()} median={int(np.median(cnt))} max={cnt.max()}')
    return (X.astype('float32'), y, classes)

def standardize_features(X: np.ndarray) -> np.ndarray:
    s = np.where(X.std(0) > 1e-08, X.std(0), 1.0)
    return ((X - X.mean(0)) / s).astype('float32')

def to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((len(y), n_classes), dtype='float32')
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y

def stratified_choice(y: np.ndarray, n_sel: int, rng) -> np.ndarray:
    n = len(y)
    n_sel = int(np.clip(n_sel, 0, n))
    sel = np.zeros(n, dtype=bool)
    if n_sel == 0:
        return sel
    classes, counts = np.unique(y, return_counts=True)
    exact = counts * (n_sel / float(n))
    k = np.floor(exact).astype(int)
    rest = n_sel - int(k.sum())
    if rest > 0:
        order = np.argsort(-(exact - k))
        for i in order[:rest]:
            k[i] += 1
    for c, kc in zip(classes, k):
        if kc <= 0:
            continue
        idx = np.where(y == c)[0]
        sel[rng.choice(idx, min(int(kc), idx.size), replace=False)] = True
    return sel

def clopper_pearson(k: int, n: int, alpha: float=0.05):
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)

def tpr_at_fpr(y, s, target: float) -> Dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype='float64')
    n_pos, n_neg = (int((y == 1).sum()), int((y == 0).sum()))
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= target + 1e-12)[0]
    i = int(ok[-1]) if ok.size else 0
    lo, hi = clopper_pearson(int(round(float(tpr[i]) * n_pos)), n_pos)
    return {'target_fpr': float(target), 'achieved_fpr': float(fpr[i]), 'tpr': float(tpr[i]), 'tpr_ci95': [lo, hi], 'reliable': bool(n_neg * target >= 10)}

def bootstrap_auc(y, s, n_boot=1000, seed=0):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype='float64')
    f, t, _ = roc_curve(y, s)
    point = float(auc(f, t))
    if n_boot <= 0:
        return (point, float('nan'), float('nan'))
    pos, neg = (np.where(y == 1)[0], np.where(y == 0)[0])
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        ff, tt, _ = roc_curve(y[idx], s[idx])
        b[i] = auc(ff, tt)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (point, float(lo), float(hi))

def summarize(y, s, fprs, n_boot, seed) -> Dict:
    a, lo, hi = bootstrap_auc(y, s, n_boot, seed)
    fpr, tpr, _ = roc_curve(y, s)
    return {'auc': a, 'auc_ci95': [lo, hi], 'significant': bool(lo > 0.5), 'balanced_acc': float(np.max(0.5 * (tpr + (1.0 - fpr)))), 'advantage': float(np.max(tpr - fpr)), 'tpr_at_fpr': [tpr_at_fpr(y, s, f) for f in fprs]}

def predict_probs(model, X, batch=8192) -> np.ndarray:
    P = np.asarray(model.predict(X, batch_size=batch, verbose=0), dtype='float64')
    if P.ndim == 1:
        P = np.stack([1.0 - P, P], axis=1)
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)

def true_class_prob(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return P[np.arange(P.shape[0]), np.asarray(y, dtype=int)]

def prob_to_phi(p_true: np.ndarray, eps: float=1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p_true, dtype='float64'), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)

def phi_to_conf(phi: np.ndarray) -> np.ndarray:
    return np.clip(expit(np.asarray(phi, dtype='float64')), 1e-12, 1.0)

def model_signals(model, X, y, batch=8192):
    P = predict_probs(model, X, batch)
    pt = true_class_prob(P, y)
    return (prob_to_phi(pt), -np.log(pt), P.argmax(axis=1))

def yeom_attack(phi_t: np.ndarray) -> np.ndarray:
    return phi_t

def lira_attack(phi_t, phi_sh, in_mask, variance='global', offline=False) -> np.ndarray:
    n = phi_sh.shape[0]
    mu_in = np.empty(n)
    mu_out = np.empty(n)
    sd_in = np.empty(n)
    sd_out = np.empty(n)
    g_in = float(np.nanstd(phi_sh[in_mask])) or 1.0
    g_out = float(np.nanstd(phi_sh[~in_mask])) or 1.0
    g_in = g_in if g_in > 1e-09 else 1.0
    g_out = g_out if g_out > 1e-09 else 1.0
    for i in range(n):
        row = phi_sh[i]
        ok = np.isfinite(row)
        vin, vout = (row[in_mask[i] & ok], row[~in_mask[i] & ok])
        mu_in[i] = vin.mean() if vin.size else np.nanmean(row)
        mu_out[i] = vout.mean() if vout.size else np.nanmean(row)
        if variance == 'per_example':
            sd_in[i] = vin.std(ddof=1) if vin.size >= 2 else g_in
            sd_out[i] = vout.std(ddof=1) if vout.size >= 2 else g_out
        else:
            sd_in[i], sd_out[i] = (g_in, g_out)
    sd_in = np.where(sd_in > 1e-09, sd_in, g_in)
    sd_out = np.where(sd_out > 1e-09, sd_out, g_out)
    if offline:
        return norm.logcdf((phi_t - mu_out) / sd_out)
    s = norm.logpdf(phi_t, mu_in, sd_in) - norm.logpdf(phi_t, mu_out, sd_out)
    fin = s[np.isfinite(s)]
    hi = fin.max() if fin.size else 0.0
    lo = fin.min() if fin.size else 0.0
    return np.nan_to_num(s, nan=lo, posinf=hi, neginf=lo)

def rmia_attack(phi_t_uni, phi_t_pop, phi_sh_uni, phi_sh_pop, in_mask_uni, a_param=0.3, gamma=1.0) -> np.ndarray:
    conf_x = phi_to_conf(phi_t_uni)
    conf_z = phi_to_conf(phi_t_pop)
    pr_out_x = np.empty(len(conf_x))
    for i in range(len(conf_x)):
        row = phi_sh_uni[i]
        out = ~in_mask_uni[i] & np.isfinite(row)
        pr_out_x[i] = phi_to_conf(row[out]).mean() if out.any() else 0.5
    pr_out_z = np.nanmean(phi_to_conf(phi_sh_pop), axis=1)
    ratio_x = conf_x / np.maximum(0.5 * ((1.0 + a_param) * pr_out_x + (1.0 - a_param)), 1e-12)
    ratio_z = conf_z / np.maximum(pr_out_z, 1e-12)
    z_sorted = np.sort(ratio_z)
    idx = np.searchsorted(z_sorted, ratio_x / gamma, side='right')
    return idx.astype('float64') / max(1, len(z_sorted))

def attack_r(phi_t, phi_sh, in_mask, midrank=True, tiebreak=True) -> np.ndarray:
    n = phi_sh.shape[0]
    p = np.empty(n)
    z = np.zeros(n)
    n_ref_min = phi_sh.shape[1]
    for i in range(n):
        ref = phi_sh[i][~in_mask[i] & np.isfinite(phi_sh[i])]
        if ref.size == 0:
            p[i] = 0.5
            continue
        n_ref_min = min(n_ref_min, ref.size)
        less = float((ref < phi_t[i]).sum())
        if midrank:
            less += 0.5 * float((ref == phi_t[i]).sum())
        p[i] = less / ref.size
        z[i] = (phi_t[i] - ref.mean()) / (ref.std() + 1e-12)
    if not tiebreak:
        return p
    step = 1.0 / (max(1, n_ref_min) + 1.0)
    return p + 0.999 * step / (1.0 + np.exp(-np.clip(z, -30, 30)))

def quantile_attack(X_uni, y_uni, phi_t_uni, X_pop, y_pop, phi_t_pop, n_classes, alpha=0.05, seed=0) -> np.ndarray:
    Fp = np.hstack([X_pop, to_one_hot(y_pop, n_classes)])
    Fu = np.hstack([X_uni, to_one_hot(y_uni, n_classes)])
    q = GradientBoostingRegressor(loss='quantile', alpha=1.0 - alpha, n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    q.fit(Fp, phi_t_pop)
    return phi_t_uni - q.predict(Fu)

def trajectory_mia_attack(traj_target, traj_shadow, member_shadow, phi_t, phi_sh_target, seed=0):
    Ftr = np.hstack([traj_shadow, phi_sh_target.reshape(-1, 1)])
    Fte = np.hstack([traj_target, phi_t.reshape(-1, 1)])
    Ftr = np.nan_to_num(Ftr, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    Fte = np.nan_to_num(Fte, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    clf.fit(Ftr, member_shadow)
    return clf.predict_proba(Fte)[:, 1]

def build_mlp(input_dim, hidden, n_classes, lr):
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([keras.Input(shape=(input_dim,))])
    for u in hidden:
        m.add(layers.Dense(u, activation='relu'))
    m.add(layers.Dense(n_classes, activation='softmax'))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss='categorical_crossentropy')
    return m

def fit_model(X, Y1h, hidden, n_classes, epochs, batch, lr):
    m = build_mlp(X.shape[1], hidden, n_classes, lr)
    m.fit(X, Y1h, epochs=epochs, batch_size=batch, shuffle=True, verbose=0)
    return m

def clear_session():
    from tensorflow import keras
    keras.backend.clear_session()

def architecture_tag(hidden):
    return 'L' + '-'.join((str(u) for u in hidden))

def compute_distillation_trajectory(teacher, X_distill, X_eval, y_eval, hidden, n_classes, epochs, batch, lr) -> np.ndarray:
    Y_soft = predict_probs(teacher, X_distill).astype('float32')
    student = build_mlp(X_distill.shape[1], hidden, n_classes, lr)
    traj = np.empty((len(X_eval), epochs))
    for e in range(epochs):
        student.fit(X_distill, Y_soft, epochs=1, batch_size=batch, shuffle=True, verbose=0)
        _, ce, _ = model_signals(student, X_eval, y_eval)
        traj[:, e] = ce
    return traj

def shadow_cache_key(hidden, n_sh, ep, batch, lr, frac, seed, n, n_classes):
    raw = f'{hidden}|{n_sh}|{ep}|{batch}|{lr}|{frac}|{seed}|{n}|{n_classes}|clf'
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def train_shadows(X_all, y_all, Y1h_all, n_uni, hidden, n_classes, n_shadows, epochs, batch, lr, in_frac, seed, cache_dir, force=False):
    n_all = len(X_all)
    key = shadow_cache_key(hidden, n_shadows, epochs, batch, lr, in_frac, seed, n_all, n_classes)
    cpath = os.path.join(cache_dir, f'{architecture_tag(hidden)}_{key}.npz')
    if os.path.exists(cpath) and (not force):
        d = np.load(cpath)
        print(f'  [shadows] reused cache: {os.path.basename(cpath)}')
        return (d['phi'], d['in_mask'])
    os.makedirs(cache_dir, exist_ok=True)
    phi = np.full((n_all, n_shadows), np.nan)
    in_mask = np.zeros((n_uni, n_shadows), dtype=bool)
    rng = np.random.default_rng(seed)
    y_uni = y_all[:n_uni]
    n_in = max(2, int(round(in_frac * n_uni)))
    t0 = time.time()
    for s in range(n_shadows):
        ts = time.time()
        sel = stratified_choice(y_uni, n_in, rng)
        in_mask[:, s] = sel
        m = fit_model(X_all[:n_uni][sel], Y1h_all[:n_uni][sel], hidden, n_classes, epochs, batch, lr)
        phi[:, s], _, _ = model_signals(m, X_all, y_all)
        clear_session()
        el = time.time() - ts
        print(f'  [shadow {s + 1:>3}/{n_shadows}] {el:5.1f}s  ETA {el * (n_shadows - s - 1) / 60:5.1f} min', flush=True)
    cov = in_mask.sum(axis=1)
    print(f'  [shadows] {(time.time() - t0) / 60:.1f} min | IN shadows per sample: min={cov.min()} median={int(np.median(cov))} max={cov.max()}')
    if cov.min() < 2 or n_shadows - cov.max() < 2:
        print('  [warning] at least one sample has fewer than two IN or OUT shadows. Increase --n_shadows.')
    np.savez_compressed(cpath, phi=phi, in_mask=in_mask)
    return (phi, in_mask)

def plot_auc_vs_capacity(results, archs, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('[plot] matplotlib is not available.')
        return
    tags = [architecture_tag(h) for h in archs]
    params = [results[t]['n_params'] for t in tags if t in results]
    tags = [t for t in tags if t in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    marks = ['o', 's', '^', 'D', 'v', 'P']
    for k, a in enumerate(ATTACKS):
        ys, los, his, xs = ([], [], [], [])
        for t, p in zip(tags, params):
            r = results[t]['attacks'].get(a)
            if r is None:
                continue
            xs.append(p)
            ys.append(r['auc'])
            los.append(r['auc'] - r['auc_ci95'][0])
            his.append(r['auc_ci95'][1] - r['auc'])
        if xs:
            ax.errorbar(xs, ys, yerr=[los, his], marker=marks[k % len(marks)], capsize=3, lw=1.4, ms=5, label=a)
    ax.axhline(0.5, color='k', ls='--', lw=0.9, label='random')
    ax.set_xscale('log')
    ax.set_xlabel('Number of model parameters (log scale)')
    ax.set_ylabel('Attack AUC')
    ax.set_title('Membership-inference vulnerability vs model capacity')
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f'[plot] {path}')

def plot_roc_grid(curves, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    tags = sorted(curves.keys())
    ncol = min(3, len(tags))
    nrow = int(math.ceil(len(tags) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    for i, t in enumerate(tags):
        ax = axes[i // ncol][i % ncol]
        for a, (y, s) in curves[t].items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(np.maximum(fpr, 1e-05), np.maximum(tpr, 1e-05), lw=1.2, label=a)
        ax.plot([1e-05, 1], [1e-05, 1], 'k--', lw=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.0001, 1)
        ax.set_ylim(0.0001, 1)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.grid(alpha=0.3, which='both', lw=0.3)
        ax.legend(fontsize=6.5)
    for j in range(len(tags), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f'[plot] {path}')

def main():
    ap = argparse.ArgumentParser(description='Membership inference attack suite for multiclass classification (softmax + cross-entropy)')
    ap.add_argument('--data_root', type=str, default=str(DATA_DIR))
    ap.add_argument('--csv', type=str, default=str(DEFAULT_DATA_FILE), help='Path to the classification dataset.')
    ap.add_argument('--label_col', type=int, default=0, help='Label column index (0 = first, -1 = last).')
    ap.add_argument('--has_header', action='store_true', help='Set when the dataset contains a header row. UCI Letter Recognition does not.')
    ap.add_argument('--universe_size', type=int, default=0, help='0 uses the full universe after reserving the auxiliary population. Smaller values reduce runtime and can increase memorization.')
    ap.add_argument('--aux_fraction', type=float, default=0.25, help='Fraction reserved as auxiliary population for RMIA and Quantile. These samples are never used for training.')
    ap.add_argument('--archs', type=str, nargs='+', default=['16', '64,32', '256,128,64', '512,256,128', '1000,1000,1000,1000'])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--in_fraction', type=float, default=0.5)
    ap.add_argument('--n_shadows', type=int, default=64)
    ap.add_argument('--variance', choices=['global', 'per_example'], default='global')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--rmia_a', type=float, default=0.3)
    ap.add_argument('--rmia_gamma', type=float, default=1.0)
    ap.add_argument('--quantile_alpha', type=float, default=0.05)
    ap.add_argument('--distill_epochs', type=int, default=30)
    ap.add_argument('--skip_tmia', action='store_true', help='Skip TMIA, the most expensive attack because it requires two distillation runs per architecture.')
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--force_shadows', action='store_true')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        given = {a.split('=')[0].lstrip('-').replace('-', '_') for a in sys.argv[1:] if a.startswith('--')}
        preset = {'n_shadows': 8, 'epochs': 40, 'universe_size': 1500, 'n_boot': 300, 'distill_epochs': 8, 'archs': ['16', '256,128,64']}
        for k, v in preset.items():
            if k not in given:
                setattr(args, k, v)
        print('[quick] fast preset enabled: results are intended for verification only.')
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    if not args.data_root and (not args.csv):
        ap.error('Provide --data_root or --csv')
    os.makedirs(args.out_dir, exist_ok=True)
    d_tgt = os.path.join(args.out_dir, 'targets')
    d_shd = os.path.join(args.out_dir, 'shadows')
    os.makedirs(d_tgt, exist_ok=True)
    os.makedirs(d_shd, exist_ok=True)
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    fprs = [0.001, 0.01, 0.05, 0.1]
    archs = [[int(x) for x in a.replace(' ', '').split(',') if x] for a in args.archs]
    print('\n' + '=' * 78 + '\n' + '1. DATA'.center(78) + '\n' + '=' * 78)
    X, y, classes = load_dataset(args.data_root, args.csv, args.label_col, args.has_header)
    n_classes = len(classes)
    X = standardize_features(X)
    print('[data] features standardized in-script; labels are class indices')
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    X, y = (X[perm], y[perm])
    n_aux = int(round(args.aux_fraction * len(X)))
    aux = stratified_choice(y, n_aux, rng)
    X_uni_all, y_uni_all = (X[~aux], y[~aux])
    X_pop, y_pop = (X[aux], y[aux])
    if args.universe_size and args.universe_size < len(X_uni_all):
        keep = stratified_choice(y_uni_all, args.universe_size, rng)
        X_uni_all, y_uni_all = (X_uni_all[keep], y_uni_all[keep])
    X = np.concatenate([X_uni_all, X_pop])
    y = np.concatenate([y_uni_all, y_pop])
    n_uni, n_aux = (len(X_uni_all), len(X_pop))
    Y1h = to_one_hot(y, n_classes)
    n_in = max(2, int(round(args.in_fraction * n_uni)))
    sel_in = stratified_choice(y[:n_uni], n_in, rng)
    idx_in = np.where(sel_in)[0]
    n_in = len(idx_in)
    member = sel_in.astype(int)
    print(f'[split] universe={n_uni} (members={n_in}, non-members={n_uni - n_in})  auxiliary_population={n_aux}')
    print('[split] class-stratified sampling keeps member and non-member class composition aligned')
    print('[split] the same membership split is reused across architectures to isolate model capacity')
    n_neg = n_uni - n_in
    usable = [f for f in fprs if n_neg * f >= 10]
    print(f'[resolution] with {n_neg} non-members, reliably estimable FPR targets: {usable}')
    cls_tr = np.unique(y[:n_uni][sel_in]).size
    if cls_tr < n_classes:
        print(f'[warning] target training data covers only {cls_tr}/{n_classes} classes; increase --universe_size or --in_fraction.')
    np.save(os.path.join(args.out_dir, 'members.npy'), member)
    np.save(os.path.join(args.out_dir, 'members_idx.npy'), idx_in)
    np.save(os.path.join(args.out_dir, 'classes.npy'), classes)
    X_uni, y_uni, Y1h_uni = (X[:n_uni], y[:n_uni], Y1h[:n_uni])
    y_pop = y[n_uni:]
    X_pop = X[n_uni:]
    Xtr, Ytr1h = (X_uni[idx_in], Y1h_uni[idx_in])
    results: Dict[str, dict] = {}
    curves: Dict[str, Dict[str, tuple]] = {}
    for ai, hidden in enumerate(archs, 1):
        tag = architecture_tag(hidden)
        print('\n' + '=' * 78)
        print(f'ARCHITECTURE {ai}/{len(archs)}: {hidden}'.center(78))
        print('=' * 78)
        print(f'[target] training ({args.epochs} epochs, no regularization)...')
        t0 = time.time()
        tgt = fit_model(Xtr, Ytr1h, hidden, n_classes, args.epochs, args.batch, args.lr)
        n_par = int(tgt.count_params())
        phi_t_uni, ce_uni, pred_uni = model_signals(tgt, X_uni, y_uni)
        phi_t_pop, ce_pop, _ = model_signals(tgt, X_pop, y_pop)
        gap = float(ce_uni[member == 0].mean() - ce_uni[idx_in].mean())
        acc_tr = float(accuracy_score(y_uni[idx_in], pred_uni[idx_in]))
        acc_ho = float(accuracy_score(y_uni[member == 0], pred_uni[member == 0]))
        print(f'[target] {time.time() - t0:.1f}s  params={n_par:,}  gap_CE={gap:.5f}  acc train={acc_tr:.4f} holdout={acc_ho:.4f}  (gap_acc={acc_tr - acc_ho:.4f})')
        if gap <= 0.0001:
            print('[warning] near-zero generalization gap: limited memorization signal for MIA evaluation.')
        try:
            tgt.save(os.path.join(d_tgt, f'{tag}.keras'))
        except Exception:
            pass
        np.save(os.path.join(d_tgt, f'{tag}.members.npy'), member)
        print(f'[shadows] training {args.n_shadows} shared shadow models for Yeom/LiRA/RMIA/Attack-R')
        phi_all, in_mask = train_shadows(X, y, Y1h, n_uni, hidden, n_classes, args.n_shadows, args.epochs, args.batch, args.lr, args.in_fraction, args.seed + 7919 * ai, d_shd, args.force_shadows)
        phi_sh_uni, phi_sh_pop = (phi_all[:n_uni], phi_all[n_uni:])
        scores: Dict[str, np.ndarray] = {}
        print('\n[attacks]')
        scores['Yeom'] = yeom_attack(phi_t_uni)
        print('  Yeom      ok (global loss threshold, no shadow models)')
        scores['LiRA'] = lira_attack(phi_t_uni, phi_sh_uni, in_mask, args.variance, args.offline)
        print('  LiRA      ok')
        scores['RMIA'] = rmia_attack(phi_t_uni, phi_t_pop, phi_sh_uni, phi_sh_pop, in_mask, args.rmia_a, args.rmia_gamma)
        print('  RMIA      ok (confidence = true-class probability)')
        scores['Attack-R'] = attack_r(phi_t_uni, phi_sh_uni, in_mask)
        print('  Attack-R  ok')
        scores['Quantile'] = quantile_attack(X_uni, y_uni, phi_t_uni, X_pop, y_pop, phi_t_pop, n_classes, args.quantile_alpha, args.seed)
        print('  Quantile  ok (conditional quantile over features and class)')
        if not args.skip_tmia:
            t1 = time.time()
            print(f'  TMIA      distilling ({args.distill_epochs} epochs x2)...', flush=True)
            rs = np.random.default_rng(args.seed + 31 * ai)
            sh_sel = stratified_choice(y_uni, n_in, rs)
            member_sh = sh_sel.astype(int)
            sh = fit_model(X_uni[sh_sel], Y1h_uni[sh_sel], hidden, n_classes, args.epochs, args.batch, args.lr)
            phi_sh_target, _, _ = model_signals(sh, X_uni, y_uni)
            traj_sh = compute_distillation_trajectory(sh, X_pop, X_uni, y_uni, hidden, n_classes, args.distill_epochs, args.batch, args.lr)
            clear_session()
            traj_tg = compute_distillation_trajectory(tgt, X_pop, X_uni, y_uni, hidden, n_classes, args.distill_epochs, args.batch, args.lr)
            scores['TMIA'] = trajectory_mia_attack(traj_tg, traj_sh, member_sh, phi_t_uni, phi_sh_target, args.seed)
            print(f'  TMIA      ok ({(time.time() - t1) / 60:.1f} min)')
        clear_session()
        res_a = {}
        curves[tag] = {}
        for a in ATTACKS:
            if a not in scores:
                continue
            r = summarize(member, scores[a], fprs, args.n_boot, args.seed)
            res_a[a] = r
            curves[tag][a] = (member, scores[a])
        results[tag] = {'hidden': hidden, 'n_params': n_par, 'gap': gap, 'acc_train': acc_tr, 'acc_holdout': acc_ho, 'attacks': res_a}
        print(f"\n  {'Attack':<11}{'AUC':>8}  {'95% CI':>18}  {'bal_acc':>8}  {'TPR@1%':>8}")
        for a, r in res_a.items():
            t1p = next((t for t in r['tpr_at_fpr'] if t['target_fpr'] == 0.01), None)
            v = f"{t1p['tpr']:.4f}" if t1p and t1p['reliable'] else 'n/e'
            star = '*' if r['significant'] else ' '
            print(f"  {a:<11}{r['auc']:>8.4f}{star} [{r['auc_ci95'][0]:.4f},{r['auc_ci95'][1]:.4f}]  {r['balanced_acc']:>8.4f}  {v:>8}")
    tags = [architecture_tag(h) for h in archs if architecture_tag(h) in results]
    W = 20 + 13 * len(tags)
    print('\n' + '=' * W)
    print('SUMMARY — AUC by attack and architecture'.center(W))
    print('=' * W)
    print(f'  universe={n_uni}  members={n_in}  auxiliary={n_aux}  classes={n_classes}  epochs={args.epochs}  shadows={args.n_shadows}')
    print('-' * W)
    print(f"{'':<20}" + ''.join((f'{t:>13}' for t in tags)))
    print(f"{'params':<20}" + ''.join((f"{results[t]['n_params']:>13,}" for t in tags)))
    print(f"{'gap CE':<20}" + ''.join((f"{results[t]['gap']:>13.4f}" for t in tags)))
    print(f"{'acc train':<20}" + ''.join((f"{results[t]['acc_train']:>13.4f}" for t in tags)))
    print(f"{'acc holdout':<20}" + ''.join((f"{results[t]['acc_holdout']:>13.4f}" for t in tags)))
    print('-' * W)
    for a in ATTACKS:
        row = f'{a:<20}'
        for t in tags:
            r = results[t]['attacks'].get(a)
            if r is None:
                row += f"{'-':>13}"
            else:
                row += f"{r['auc']:>12.4f}" + ('*' if r['significant'] else ' ')
        print(row)
    print('-' * W)
    print('  * = the 95% AUC confidence interval excludes 0.5.')
    print('=' * W)
    print('\n' + '=' * W)
    print('TPR @ FPR=1%'.center(W))
    print('=' * W)
    print(f"{'':<20}" + ''.join((f'{t:>13}' for t in tags)))
    print('-' * W)
    for a in ATTACKS:
        row = f'{a:<20}'
        for t in tags:
            r = results[t]['attacks'].get(a)
            if r is None:
                row += f"{'-':>13}"
            else:
                t1 = next((x for x in r['tpr_at_fpr'] if x['target_fpr'] == 0.01), None)
                row += f"{t1['tpr']:>13.4f}" if t1 and t1['reliable'] else f"{'n/e':>13}"
        print(row)
    print('=' * W)
    print('\nTREND: model capacity -> membership leakage')
    print('-' * 60)
    pars = np.array([results[t]['n_params'] for t in tags], dtype=float)
    gaps = np.array([results[t]['gap'] for t in tags], dtype=float)
    for a in ATTACKS:
        aucs = [results[t]['attacks'][a]['auc'] for t in tags if a in results[t]['attacks']]
        if len(aucs) < 3:
            continue
        aucs = np.array(aucs)
        mono = bool(np.all(np.diff(aucs) >= -1e-09))
        cg = float(np.corrcoef(gaps[:len(aucs)], aucs)[0, 1])
        cp = float(np.corrcoef(np.log10(pars[:len(aucs)]), aucs)[0, 1])
        print(f"  {a:<11} corr(gap,AUC)={cg:+.3f}  corr(log params,AUC)={cp:+.3f}  monotonic={('yes' if mono else 'NO')}")
    print('-' * 60)
    print("  If monotonic=NO, leakage tracks the generalization gap more closely than")
    print('  raw parameter count; report that relationship rather than assuming leakage')
    print('  necessarily increases with model size.')
    plot_auc_vs_capacity(results, archs, os.path.join(args.out_dir, 'auc_vs_capacity.png'))
    plot_roc_grid(curves, os.path.join(args.out_dir, 'roc_grid.png'))
    with open(os.path.join(args.out_dir, 'mia_results.json'), 'w') as f:
        json.dump({'config': vars(args), 'n_classes': int(n_classes), 'results': results}, f, indent=2, default=str)
    with open(os.path.join(args.out_dir, 'mia_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arch', 'n_params', 'gap_ce', 'acc_train', 'acc_holdout', 'attack', 'auc', 'auc_lo', 'auc_hi', 'significant', 'balanced_acc', 'advantage'] + [f'tpr@{x}' for x in fprs])
        for t in tags:
            r0 = results[t]
            for a, r in r0['attacks'].items():
                w.writerow([t, r0['n_params'], r0['gap'], r0['acc_train'], r0['acc_holdout'], a, r['auc'], r['auc_ci95'][0], r['auc_ci95'][1], r['significant'], r['balanced_acc'], r['advantage']] + [x['tpr'] if x['reliable'] else '' for x in r['tpr_at_fpr']])
    per_attack_dir = os.path.join(args.out_dir, 'per_attack')
    os.makedirs(per_attack_dir, exist_ok=True)
    written = []
    for a in ATTACKS:
        rows = [(t, results[t]) for t in tags if a in results[t]['attacks']]
        if not rows:
            continue
        fname = os.path.join(per_attack_dir, f"{a.replace('-', '_').lower()}.csv")
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            head = ['arch', 'hidden', 'n_params', 'gap_ce', 'acc_train', 'acc_holdout', 'auc', 'auc_lo', 'auc_hi', 'significant', 'balanced_acc', 'advantage', 'n_members', 'n_nonmembers']
            for x in fprs:
                head += [f'tpr@{x}', f'tpr@{x}_lo', f'tpr@{x}_hi', f'tpr@{x}_reliable']
            w.writerow(head)
            for t, r0 in rows:
                r = r0['attacks'][a]
                row = [t, '-'.join(map(str, r0['hidden'])), r0['n_params'], r0['gap'], r0['acc_train'], r0['acc_holdout'], r['auc'], r['auc_ci95'][0], r['auc_ci95'][1], r['significant'], r['balanced_acc'], r['advantage'], n_in, n_uni - n_in]
                for x in r['tpr_at_fpr']:
                    row += [x['tpr'], x['tpr_ci95'][0], x['tpr_ci95'][1], True] if x['reliable'] else ['', '', '', False]
                w.writerow(row)
        written.append(os.path.basename(fname))
    print(f"[csv] one file per attack in {per_attack_dir}/: {', '.join(written)}")
    for metric, fname in [('auc', 'auc_matrix.csv'), ('tpr01', 'tpr_1pct_matrix.csv')]:
        with open(os.path.join(args.out_dir, fname), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['attack'] + tags)
            for a in ATTACKS:
                row = [a]
                for t in tags:
                    r = results[t]['attacks'].get(a)
                    if r is None:
                        row.append('')
                    elif metric == 'auc':
                        row.append(r['auc'])
                    else:
                        x = next((z for z in r['tpr_at_fpr'] if z['target_fpr'] == 0.01), None)
                        row.append(x['tpr'] if x and x['reliable'] else '')
                w.writerow(row)
    print(f'\nOutputs in {args.out_dir}/: mia_results.json, mia_summary.csv, per_attack/, auc_matrix.csv, tpr_1pct_matrix.csv, auc_vs_capacity.png, roc_grid.png, members.npy, classes.npy, targets/, shadows/')
if __name__ == '__main__':
    main()
