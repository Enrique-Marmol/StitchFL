# -*- coding: utf-8 -*-
"""Federated LSTM benchmark for heterogeneous client capacities.

Compares local full-model training, position-aware heterogeneous federated
learning, and FedAvg on private client time-series datasets. The final global
models are evaluated independently on each client's held-out test set.
"""

import os
import csv
import pathlib
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tabulate import tabulate

tf.random.set_seed(42)
np.random.seed(42)


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[GPU] {len(gpus)} GPU(s) detected: "
                  f"{[g.name for g in gpus]}  (memory growth ON)")
        except RuntimeError as e:
            print(f"[GPU] Could not configure memory growth: {e}")
    else:
        print("[GPU] No GPU detected; running on CPU.")

configure_gpu()


BASE_DIR = pathlib.Path(os.getcwd())
DATA_DIR = BASE_DIR / "Datasets" / "pleiadata" 
CLIENT_CSV_PATHS = [
    DATA_DIR / "data-model-consumoA-60T.csv",
    DATA_DIR / "data-model-consumoB-60T.csv",
    DATA_DIR / "data-model-consumoC-60T.csv",
]
TARGET = "dif_cons_smooth"
LOOKBACK = 24
SERVER_HIDDEN_LAYERS = [64, 32]
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

FL_ROUNDS = 10
FL_LOCAL_EPOCHS = 5
FL_BATCH_SIZE = BATCH_SIZE
FL_VERBOSE_ROUND_EVAL = True
CALIBRATE_HEAD_EVERY_ROUND = False

CENTRAL_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15

NUM_ANCHORS = 1000
ANCHOR_RHO = 0.7
ANCHOR_ALPHA = 1e-2

NUM_CLIENTS = len(CLIENT_CSV_PATHS)
CLIENT_FRACTION = 0.56


def make_overlapping_ranges(hidden_size, n_clients, frac):
    win = min(hidden_size, max(1, int(round(frac * hidden_size))))
    if n_clients == 1:
        return [(0, hidden_size)]
    starts = np.linspace(0, hidden_size - win, n_clients).round().astype(int)
    return [(int(s), int(s + win)) for s in starts]


_ranges_per_layer = [
    make_overlapping_ranges(hidden_size, NUM_CLIENTS, CLIENT_FRACTION)
    for hidden_size in SERVER_HIDDEN_LAYERS
]
CLIENT_LAYER_RANGES = [
    [_ranges_per_layer[li][c] for li in range(len(SERVER_HIDDEN_LAYERS))]
    for c in range(NUM_CLIENTS)
]

SEED = 42


RESULTS_DIR = BASE_DIR / "results" / "LSTM"

FUSION_RESULTS_CSV = RESULTS_DIR / "stitchfl.csv"
FEDAVG_RESULTS_CSV = RESULTS_DIR / "fedavg.csv"

SAVED_METRICS = ["RMSE", "MAE", "R2", "NMBE_pct", "CV_RMSE_pct"]


def append_metrics_row(path, metric_summary, keys=SAVED_METRICS):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "architecture", "n_clients",
              "rounds", "local_epochs"] + keys
    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "-".join(str(u) for u in SERVER_HIDDEN_LAYERS),
        "n_clients":    NUM_CLIENTS,
        "rounds":       FL_ROUNDS,
        "local_epochs": FL_LOCAL_EPOCHS,
    }
    for k in keys:
        row[k] = round(metric_summary[k][0], 6)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"    -> saved to {path}")


def load_time_series_csv(path):
    try:
        df = pd.read_csv(path, sep=';', index_col=0)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            return df.sort_values('Date').reset_index(drop=True)
    except Exception:
        pass
    df = pd.read_csv(path, sep=';')
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
    date_col = next((c for c in df.columns
                     if c.lower() in ('date', 'fecha', 'datetime', 'timestamp')), None)
    if date_col is None and df.index.name and df.index.name.lower() in ('date', 'fecha'):
        df = df.reset_index().rename(columns={df.index.name: 'Date'})
    elif date_col:
        df = df.rename(columns={date_col: 'Date'})
    else:
        raise KeyError(f"Date column not found. Columns: {df.columns.tolist()}")
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    return df.sort_values('Date').reset_index(drop=True)


def build_features(df):
    hour = df['Date'].dt.hour
    day_of_year  = df['Date'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['doy_sin']  = np.sin(2 * np.pi * day_of_year / 365)
    df['doy_cos']  = np.cos(2 * np.pi * day_of_year / 365)
    df['cons_lag24']  = df[TARGET].shift(24)
    df['cons_lag168'] = df[TARGET].shift(168)
    df = df.dropna().reset_index(drop=True)

    EXCLUDED_COLUMNS = ['Date', 'dif_cons_real', 'cons_total',
            'Hour_1', 'Hour_2', 'Hour_3',
            'Season_1', 'Season_2', 'Season_3', 'Season_4']
    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLUMNS + [TARGET]]
    return df, feature_cols


def split_and_scale(df, feature_cols):
    data = df[feature_cols + [TARGET]].values.astype(np.float32)
    target_idx = data.shape[1] - 1
    n = len(data)
    n_tr, n_va = int(n * 0.80), int(n * 0.10)
    tr_raw = data[:n_tr]
    va_raw = data[n_tr:n_tr + n_va]
    te_raw = data[n_tr + n_va:]
    scaler = StandardScaler().fit(tr_raw)
    tr = scaler.transform(tr_raw)
    va = scaler.transform(va_raw)
    te = scaler.transform(te_raw)
    return dict(tr=tr, va=va, te=te,
                tr_raw=tr_raw, va_raw=va_raw, te_raw=te_raw,
                scaler=scaler, target_idx=target_idx,
                tgt_mu=scaler.mean_[target_idx],
                tgt_sd=scaler.scale_[target_idx])


def windows(arr, lb, ti):
    X = np.stack([arr[i:i + lb] for i in range(len(arr) - lb)])
    y = arr[lb:, ti]
    return X.astype(np.float32), y.astype(np.float32)


def load_client_data(csv_path):
    df = load_time_series_csv(csv_path)
    df, feature_cols = build_features(df)
    sp = split_and_scale(df, feature_cols)
    ti = sp["target_idx"]
    X_tr, y_tr = windows(sp["tr"], LOOKBACK, ti)
    X_va, y_va = windows(sp["va"], LOOKBACK, ti)
    X_te, y_te = windows(sp["te"], LOOKBACK, ti)
    return dict(
        X_tr=X_tr, y_tr=y_tr,
        X_va=X_va, y_va=y_va,
        X_te=X_te, y_te=y_te,
        tgt_mu=sp["tgt_mu"], tgt_sd=sp["tgt_sd"],
        y_train_range=float(sp["tr_raw"][:, ti].max() - sp["tr_raw"][:, ti].min()),
        feature_cols=feature_cols,
        n_train=len(X_tr), n_val=len(X_va), n_test=len(X_te),
        name=pathlib.Path(csv_path).stem,
    )


def build_lstm_stack(hidden_sizes, input_dim):
    layers = [Input(shape=(LOOKBACK, input_dim))]
    n = len(hidden_sizes)
    for li, h in enumerate(hidden_sizes):
        layers.append(LSTM(h, return_sequences=(li < n - 1)))
        layers.append(Dropout(DROPOUT))
    layers.append(Dense(1))
    m = Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse', metrics=['mae'])
    return m


def build_server_model(input_dim):
    return build_lstm_stack(SERVER_HIDDEN_LAYERS, input_dim)


def build_client_model(layer_sizes, input_dim):
    return build_lstm_stack(layer_sizes, input_dim)


def _get_lstms(model):
    return [l for l in model.layers if isinstance(l, LSTM)]


def _get_dense(model):
    return [l for l in model.layers if isinstance(l, Dense)][0]


def _gate_cols(cols, hidden_size):
    # Keras concatenates the four LSTM gates along the last dimension.
    return np.concatenate([cols + g * hidden_size for g in range(4)])


def get_model_weights(model):
    global_weights = {'W': [], 'U': [], 'b': []}
    for lstm in _get_lstms(model):
        k, U, b = [w.astype(np.float32) for w in lstm.get_weights()]
        global_weights['W'].append(k); global_weights['U'].append(U); global_weights['b'].append(b)
    Wd, bd = [w.astype(np.float32) for w in _get_dense(model).get_weights()]
    global_weights['Wd'] = Wd; global_weights['bd'] = bd
    return global_weights


def build_client_from_weights(global_weights, layer_ranges, input_dim):
    sizes = [e - s for (s, e) in layer_ranges]
    m = build_client_model(sizes, input_dim)
    lstms = _get_lstms(m)
    dense = _get_dense(m)
    for li, lstm_c in enumerate(lstms):
        hidden_size = SERVER_HIDDEN_LAYERS[li]
        s, e = layer_ranges[li]
        cols = np.arange(s, e)
        gc = _gate_cols(cols, hidden_size)
        if li == 0:
            k_c = global_weights['W'][li][:, gc].copy()
        else:
            s_prev, e_prev = layer_ranges[li - 1]
            rows_prev = np.arange(s_prev, e_prev)
            k_c = global_weights['W'][li][np.ix_(rows_prev, gc)].copy()
        U_c = global_weights['U'][li][np.ix_(cols, gc)].copy()
        b_c = global_weights['b'][li][gc].copy()
        lstm_c.set_weights([k_c, U_c, b_c])
    s_last, e_last = layer_ranges[-1]
    rows_last = np.arange(s_last, e_last)
    dense.set_weights([global_weights['Wd'][rows_last, :].copy(), global_weights['bd'].copy()])
    return m


def fuse_clients_into(server, clients, client_layer_ranges, input_dim, fallback):
    # Average covered positions and preserve previous global weights elsewhere.
    n_layers = len(SERVER_HIDDEN_LAYERS)

    W_sum, W_cnt, U_sum, U_cnt, b_sum, b_cnt = [], [], [], [], [], []
    for li, hidden_size in enumerate(SERVER_HIDDEN_LAYERS):
        in_dim = input_dim if li == 0 else SERVER_HIDDEN_LAYERS[li - 1]
        W_sum.append(np.zeros((in_dim, 4 * hidden_size), np.float32))
        W_cnt.append(np.zeros((in_dim, 4 * hidden_size), np.float32))
        U_sum.append(np.zeros((hidden_size, 4 * hidden_size), np.float32))
        U_cnt.append(np.zeros((hidden_size, 4 * hidden_size), np.float32))
        b_sum.append(np.zeros((4 * hidden_size,), np.float32))
        b_cnt.append(np.zeros((4 * hidden_size,), np.float32))

    last_hidden_size = SERVER_HIDDEN_LAYERS[-1]
    Wd_sum = np.zeros((last_hidden_size, 1), np.float32); Wd_cnt = np.zeros((last_hidden_size, 1), np.float32)
    bd_sum = np.zeros((1,), np.float32);        bd_cnt = np.zeros((1,), np.float32)

    for c, layer_ranges in zip(clients, client_layer_ranges):
        lstms = _get_lstms(c)
        dense = _get_dense(c)
        for li, lstm in enumerate(lstms):
            hidden_size = SERVER_HIDDEN_LAYERS[li]
            s, e = layer_ranges[li]
            cols = np.arange(s, e)
            gc = _gate_cols(cols, hidden_size)
            k, U, b = lstm.get_weights()

            if li == 0:

                W_sum[li][:, gc] += k.astype(np.float32)
                W_cnt[li][:, gc] += 1.0
            else:

                s_prev, e_prev = layer_ranges[li - 1]
                rows_prev = np.arange(s_prev, e_prev)
                W_sum[li][np.ix_(rows_prev, gc)] += k.astype(np.float32)
                W_cnt[li][np.ix_(rows_prev, gc)] += 1.0

            U_sum[li][np.ix_(cols, gc)] += U.astype(np.float32)
            U_cnt[li][np.ix_(cols, gc)] += 1.0
            b_sum[li][gc] += b.astype(np.float32)
            b_cnt[li][gc] += 1.0


        s_last, e_last = layer_ranges[-1]
        rows_last = np.arange(s_last, e_last)
        Wd, bd = dense.get_weights()
        Wd_sum[rows_last, :] += Wd.astype(np.float32)
        Wd_cnt[rows_last, :] += 1.0
        bd_sum += bd.astype(np.float32)
        bd_cnt += 1.0

    def sd(n, d, fb):
        avg = n / np.where(d == 0, 1.0, d)
        return np.where(d == 0, fb.astype(np.float32), avg).astype(np.float32)

    coverage = {}
    for li in range(n_layers):
        coverage[f"L{li}_kernel"]    = round(float((W_cnt[li] > 0).mean()), 3)
        coverage[f"L{li}_recurrent"] = round(float((U_cnt[li] > 0).mean()), 3)
    coverage["dense_kernel"] = round(float((Wd_cnt > 0).mean()), 3)

    for li, lstm_s in enumerate(_get_lstms(server)):
        lstm_s.set_weights([
            sd(W_sum[li], W_cnt[li], fallback['W'][li]),
            sd(U_sum[li], U_cnt[li], fallback['U'][li]),
            sd(b_sum[li], b_cnt[li], fallback['b'][li]),
        ])
    _get_dense(server).set_weights([
        sd(Wd_sum, Wd_cnt, fallback['Wd']),
        sd(bd_sum, bd_cnt, fallback['bd']),
    ])
    return coverage


def make_synthetic_anchors(n_anchors, lookback, input_dim, rho, seed):
    # AR(1) anchors approximate standardized temporal inputs without private data.
    rng = np.random.RandomState(seed)
    X = np.zeros((n_anchors, lookback, input_dim), dtype=np.float32)
    X[:, 0, :] = rng.randn(n_anchors, input_dim).astype(np.float32)
    sigma = np.sqrt(1.0 - rho ** 2)
    for t in range(1, lookback):
        X[:, t, :] = rho * X[:, t - 1, :] + sigma * rng.randn(n_anchors, input_dim).astype(np.float32)
    return X


def calibrate_server_head(server, clients, input_dim,
                          n_anchors=NUM_ANCHORS, rho=ANCHOR_RHO, alpha=ANCHOR_ALPHA):

    X_anc = make_synthetic_anchors(n_anchors, LOOKBACK, input_dim, rho, SEED)


    preds = np.stack(
        [c.predict(X_anc, verbose=0).ravel() for c in clients], axis=0
    )
    y_target = preds.mean(axis=0)


    last_lstm = _get_lstms(server)[-1]
    dense_s   = _get_dense(server)
    feat_extractor = tf.keras.Model(inputs=server.inputs, outputs=last_lstm.output)
    h_s = feat_extractor.predict(X_anc, verbose=0)


    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(h_s, y_target)
    W_new = ridge.coef_.reshape(SERVER_HIDDEN_LAYERS[-1], 1).astype(np.float32)
    b_new = np.array([ridge.intercept_], dtype=np.float32)

    dense_s.set_weights([W_new, b_new])
    return server


def full_metrics(y, yhat, y_train_range=None):
    err = yhat - y
    ae  = np.abs(err)
    mean_y = np.mean(y)
    nz = np.abs(y) > 1e-6
    denom = (np.abs(y) + np.abs(yhat)) / 2
    smape_mask = denom > 1e-8
    out = dict(
        MAE  = float(mean_absolute_error(y, yhat)),
        MSE  = float(mean_squared_error(y, yhat)),
        RMSE = float(np.sqrt(mean_squared_error(y, yhat))),
        R2   = float(r2_score(y, yhat)),
        MAPE = float(np.mean(ae[nz] / np.abs(y[nz])) * 100) if nz.any() else float('nan'),
        SMAPE= float(np.mean(ae[smape_mask] / denom[smape_mask]) * 100) if smape_mask.any() else float('nan'),
        Pearson_r = float(np.corrcoef(y, yhat)[0, 1]),
        ME_bias   = float(err.mean()),
        NMBE_pct  = float(err.mean() / mean_y * 100) if abs(mean_y) > 1e-8 else float('nan'),
        MedAE     = float(np.median(ae)),
        Max_AE    = float(np.max(ae)),
        CV_RMSE_pct = float(np.sqrt(mean_squared_error(y, yhat)) / mean_y * 100) if abs(mean_y) > 1e-8 else float('nan'),
    )
    if y_train_range is not None and y_train_range > 0:
        out["NRMSE_pct"] = float(out["RMSE"] / y_train_range * 100)
    return out


def aggregate_metrics(dicts, keys):
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in dicts if not np.isnan(d.get(k, np.nan))])
        if len(vals) == 0:
            out[k] = (float('nan'), float('nan'))
        else:
            out[k] = (float(vals.mean()), float(vals.std()))
    return out


def predict_real(model, X, tgt_mu, tgt_sd):
    yp_s = model.predict(X, verbose=0, batch_size=256).ravel()
    return yp_s * tgt_sd + tgt_mu


def evaluate(model, X_te, y_te_s, tgt_mu, tgt_sd, y_train_range):
    y_pred = predict_real(model, X_te, tgt_mu, tgt_sd)
    y_true = y_te_s * tgt_sd + tgt_mu
    return full_metrics(y_true, y_pred, y_train_range)


def evaluate_across_clients(model, client_datasets, label):
    metric_records = []
    for i, client_data in enumerate(client_datasets, 1):
        client_metrics = evaluate(
            model,
            client_data["X_te"],
            client_data["y_te"],
            client_data["tgt_mu"],
            client_data["tgt_sd"],
            client_data["y_train_range"],
        )
        print(
            f"    {label} on test C{i} ({client_data['name']}): "
            f"R²={client_metrics['R2']:+.4f}  MAE={client_metrics['MAE']:.3f}  "
            f"RMSE={client_metrics['RMSE']:.3f}  "
            f"CV(RMSE)={client_metrics['CV_RMSE_pct']:.2f}%"
        )
        metric_records.append(client_metrics)
    return metric_records


def mean_validation_r2(model, client_datasets):
    r2s = []
    for client_data in client_datasets:
        y_pred = predict_real(model, client_data["X_va"], client_data["tgt_mu"], client_data["tgt_sd"])
        y_true = client_data["y_va"] * client_data["tgt_sd"] + client_data["tgt_mu"]
        r2s.append(r2_score(y_true, y_pred))
    return float(np.mean(r2s))


def print_round_progress(method, round_id, total_rounds, model, client_datasets):
    if not FL_VERBOSE_ROUND_EVAL:
        print(f"    [{method}] Round {round_id}/{total_rounds} | fusion applied "
              f"at the server")
        return
    r2 = mean_validation_r2(model, client_datasets)
    print(f"    [{method}] Round {round_id}/{total_rounds} | fusion applied "
          f"| mean validation R² across clients = {r2:+.4f}")


def train_local_reference(client_data):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    model = build_server_model(client_data["X_tr"].shape[2])
    cbs = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.5,
                          min_lr=1e-5, verbose=0),
    ]
    model.fit(client_data["X_tr"], client_data["y_tr"],
              validation_data=(client_data["X_va"], client_data["y_va"]),
              epochs=CENTRAL_EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
    return model


def run_local_references(client_datasets):
    metric_records = []
    for i, client_data in enumerate(client_datasets, 1):
        model = train_local_reference(client_data)
        client_metrics = evaluate(
            model,
            client_data["X_te"],
            client_data["y_te"],
            client_data["tgt_mu"],
            client_data["tgt_sd"],
            client_data["y_train_range"],
        )
        print(
            f"    Reference C{i} ({client_data['name']})  "
            f"train={client_data['n_train']:5d}  test={client_data['n_test']:5d}  "
            f"R²={client_metrics['R2']:+.4f}  MAE={client_metrics['MAE']:.3f}  "
            f"RMSE={client_metrics['RMSE']:.3f}  "
            f"CV(RMSE)={client_metrics['CV_RMSE_pct']:.2f}%"
        )
        metric_records.append(client_metrics)
        del model
        tf.keras.backend.clear_session()
    return metric_records


def run_heterofl(client_datasets):
    ref = client_datasets[0]
    input_dim = ref["X_tr"].shape[2]


    for i, client_data in enumerate(client_datasets, 1):
        if client_data["X_tr"].shape[2] != input_dim:
            raise ValueError(
                f"Client {i} ({client_data['name']}) has input_dim={client_data['X_tr'].shape[2]} "
                f"!= reference {input_dim}. Feature dimensions must match across client CSVs."
            )


    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    server = build_server_model(input_dim)
    global_weights = get_model_weights(server)

    clients = []
    coverage = {}
    for round_id in range(1, FL_ROUNDS + 1):
        print(f"\n    {'=' * 10} ROUND HETERO-FL {round_id}/{FL_ROUNDS} {'=' * 10}")


        for c in clients:
            del c
        clients = []
        gc.collect()

        for i, (client_data, layer_ranges) in enumerate(zip(client_datasets, CLIENT_LAYER_RANGES), 1):
            np.random.seed(SEED + i + 1000 * round_id)
            tf.random.set_seed(SEED + i + 1000 * round_id)

            m = build_client_from_weights(global_weights, layer_ranges, input_dim)

            hist = m.fit(client_data["X_tr"], client_data["y_tr"],
                         validation_data=(client_data["X_va"], client_data["y_va"]),
                         epochs=FL_LOCAL_EPOCHS, batch_size=FL_BATCH_SIZE, verbose=0)
            sizes_str = "+".join(str(e - s) for (s, e) in layer_ranges)
            print(f"      C{i} ({client_data['name']})  layers={sizes_str:9s}  "
                  f"loss={hist.history['loss'][-1]:.5f}  "
                  f"val_loss={hist.history['val_loss'][-1]:.5f}")
            clients.append(m)


        coverage = fuse_clients_into(server, clients, CLIENT_LAYER_RANGES, input_dim,
                                 fallback=global_weights)
        if CALIBRATE_HEAD_EVERY_ROUND:
            calibrate_server_head(server, clients, input_dim)
        global_weights = get_model_weights(server)
        print_round_progress("Hetero-FL", round_id, FL_ROUNDS, server, client_datasets)

    print(f"    coverage (last round): {coverage}")


    client_metrics = []
    for i, (m, client_data) in enumerate(zip(clients, client_datasets), 1):
        metrics = evaluate(m, client_data["X_te"], client_data["y_te"],
                      client_data["tgt_mu"], client_data["tgt_sd"],
                      client_data["y_train_range"])
        print(f"    Submodel C{i} ({client_data['name']}) on its test: R²={metrics['R2']:+.4f}  "
              f"MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
              f"CV(RMSE)={metrics['CV_RMSE_pct']:.2f}%")
        client_metrics.append(metrics)


    if not CALIBRATE_HEAD_EVERY_ROUND:
        calibrate_server_head(server, clients, input_dim)


    fusion_metrics = evaluate_across_clients(server, client_datasets, "Fusion")

    for c in clients:
        del c
    del server
    tf.keras.backend.clear_session()
    gc.collect()

    return client_metrics, fusion_metrics


def run_fedavg(client_datasets):
    input_dim = client_datasets[0]["X_tr"].shape[2]
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    global_model = build_server_model(input_dim)
    local_model  = build_server_model(input_dim)

    counts = [client_data["n_train"] for client_data in client_datasets]
    total  = float(sum(counts))
    client_weights  = [n / total for n in counts]

    for round_id in range(1, FL_ROUNDS + 1):
        print(f"\n    {'=' * 10} ROUND FEDAVG {round_id}/{FL_ROUNDS} {'=' * 10}")
        global_weights_list = global_model.get_weights()
        acc = [np.zeros_like(w) for w in global_weights_list]

        for i, (client_data, client_weight) in enumerate(zip(client_datasets, client_weights), 1):

            local_model.set_weights(global_weights_list)

            local_model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                                loss='mse', metrics=['mae'])

            hist = local_model.fit(client_data["X_tr"], client_data["y_tr"],
                                   validation_data=(client_data["X_va"], client_data["y_va"]),
                                   epochs=FL_LOCAL_EPOCHS, batch_size=FL_BATCH_SIZE,
                                   verbose=0)

            for k, w in enumerate(local_model.get_weights()):
                acc[k] += client_weight * w
            print(f"      C{i} ({client_data['name']})  full model {SERVER_HIDDEN_LAYERS}  "
                  f"n={client_data['n_train']:5d}  weight={client_weight:.4f}  "
                  f"loss={hist.history['loss'][-1]:.5f}  "
                  f"val_loss={hist.history['val_loss'][-1]:.5f}")


        global_model.set_weights(acc)
        print_round_progress("FedAvg", round_id, FL_ROUNDS, global_model, client_datasets)


    fedavg_metrics = evaluate_across_clients(global_model, client_datasets, "FedAvg")

    del local_model, global_model
    tf.keras.backend.clear_session()
    gc.collect()
    return fedavg_metrics


def main():
    print("[1/5] Loading and preprocessing client datasets...")
    client_datasets = []
    for i, csv_path in enumerate(CLIENT_CSV_PATHS, 1):
        client_data = load_client_data(csv_path)
        client_datasets.append(client_data)
        print(f"  C{i}: {client_data['name']}  train={client_data['n_train']:5d}  val={client_data['n_val']:4d}  "
              f"test={client_data['n_test']:4d}  features={len(client_data['feature_cols'])}")

    print(f"\n  Server configuration : LSTM stack {SERVER_HIDDEN_LAYERS} -> Dropout -> Dense(1)")
    print(f"  Federated configuration : {FL_ROUNDS} rounds x {FL_LOCAL_EPOCHS} local epochs "
          f"| server fusion after every round | final evaluation only")
    print(f"  Hetero-FL configuration: {NUM_CLIENTS} clients, fraction/layer ≈ {CLIENT_FRACTION:.2f}")
    for i, layer_ranges in enumerate(CLIENT_LAYER_RANGES, 1):
        desc = "  ".join(f"L{li}[{s:2d}..{e:2d})({e-s}u)"
                         for li, (s, e) in enumerate(layer_ranges))
        print(f"     C{i}: {desc}")


    print(f"\n[2/5] REFERENCE — full model {SERVER_HIDDEN_LAYERS} trained "
          f"independently on each client and evaluated on its own test set...")
    reference_metrics = run_local_references(client_datasets)


    print(f"\n[3/5] Hetero-FL — {FL_ROUNDS} rounds x {FL_LOCAL_EPOCHS} local epochs; "
          f"server fusion after every round; final fused model evaluated on every "
          f"client test set...")
    submodel_metrics, fusion_metrics = run_heterofl(client_datasets)


    print(f"\n[4/5] FedAvg — {FL_ROUNDS} rounds x {FL_LOCAL_EPOCHS} local epochs "
          f"with the full model on every client; final model evaluated on every "
          f"client test set...")
    fedavg_metrics = run_fedavg(client_datasets)


    keys_all = ["MAE", "MSE", "RMSE", "R2", "MAPE", "SMAPE", "Pearson_r",
                "ME_bias", "NMBE_pct", "MedAE", "Max_AE",
                "CV_RMSE_pct", "NRMSE_pct"]

    reference_summary = aggregate_metrics(reference_metrics, keys_all)
    submodel_summary     = aggregate_metrics(submodel_metrics, keys_all)
    fusion_summary  = aggregate_metrics(fusion_metrics, keys_all)
    fedavg_summary  = aggregate_metrics(fedavg_metrics, keys_all)


    print("\n[5/5] Saving mean metrics (RMSE, MAE, R2, NMBE, CV_RMSE) to CSV...")
    append_metrics_row(REFERENCE_RESULTS_CSV,    reference_summary)
    append_metrics_row(FUSION_RESULTS_CSV, fusion_summary)
    append_metrics_row(FEDAVG_RESULTS_CSV, fedavg_summary)

    print("\n" + "=" * 118)
    print(f"FULL COMPARISON — mean ± std across all client test sets "
          f"({FL_ROUNDS} rounds x {FL_LOCAL_EPOCHS} epochs)")
    print("  Reference = full model trained locally on each client and evaluated on its own test")
    print("  Fusion    = round-based Hetero-FL; fused global model evaluated on each test set")
    print("  FedAvg    = full model trained with FedAvg and evaluated on each client test")
    print("=" * 118)
    rows = []
    for k in keys_all:
        rows.append({
            "Metric":                    k,
            "Reference (mean)":         f"{reference_summary[k][0]:+.4f} ± {reference_summary[k][1]:.4f}",
            "Client submodels (mean)":  f"{submodel_summary[k][0]:+.4f} ± {submodel_summary[k][1]:.4f}",
            "Hetero-FL fusion (mean)":   f"{fusion_summary[k][0]:+.4f} ± {fusion_summary[k][1]:.4f}",
            "FedAvg (mean)":             f"{fedavg_summary[k][0]:+.4f} ± {fedavg_summary[k][1]:.4f}",
        })
    print(tabulate(rows, headers="keys", tablefmt="github"))

    print("\n" + "=" * 118)
    print("PER-CLIENT RESULTS — R², MAE, RMSE, CV(RMSE), SMAPE")
    print("=" * 118)
    summary_rows = []
    for i, (client_data, cm, sm, fm, am) in enumerate(
            zip(client_datasets, reference_metrics, submodel_metrics, fusion_metrics, fedavg_metrics), 1):
        for label, metrics in (
            (f"C{i} ({client_data['name']}) — Reference (full model local)", cm),
            (f"C{i} ({client_data['name']}) — local submodel (after rounds)",     sm),
            (f"C{i} ({client_data['name']}) — Hetero-FL fusion",                   fm),
            (f"C{i} ({client_data['name']}) — FedAvg",                             am),
        ):
            summary_rows.append({
                "Model": label,
                "R²": f"{metrics['R2']:+.4f}", "MAE": f"{metrics['MAE']:.3f}",
                "RMSE": f"{metrics['RMSE']:.3f}", "CV(RMSE)%": f"{metrics['CV_RMSE_pct']:.2f}",
                "SMAPE%": f"{metrics['SMAPE']:.2f}", "NMBE%": f"{metrics['NMBE_pct']:+.2f}"})
    print(tabulate(summary_rows, headers="keys", tablefmt="github"))

    print("\n" + "=" * 118)
    print("SUMMARY — MEANS (Reference  vs  Hetero-FL fusion  vs  FedAvg)")
    print("=" * 118)

    def mean_row(label, metric_summary):
        return {"Model": label,
                "R²":        f"{metric_summary['R2'][0]:+.4f} ± {metric_summary['R2'][1]:.4f}",
                "MAE":       f"{metric_summary['MAE'][0]:.3f}",
                "RMSE":      f"{metric_summary['RMSE'][0]:.3f}",
                "CV(RMSE)%": f"{metric_summary['CV_RMSE_pct'][0]:.2f}",
                "SMAPE%":    f"{metric_summary['SMAPE'][0]:.2f}",
                "NMBE%":     f"{metric_summary['NMBE_pct'][0]:+.2f}"}

    mean_rows = [
        mean_row(f"Reference (full model local) — mean {NUM_CLIENTS} clients", reference_summary),
        mean_row(f"Hetero-FL fusion — mean {NUM_CLIENTS} clients", fusion_summary),
        mean_row(f"FedAvg — mean {NUM_CLIENTS} clients", fedavg_summary),
    ]
    print(tabulate(mean_rows, headers="keys", tablefmt="github"))


    def delta(metric_summary, base, label):
        dR2   = metric_summary['R2'][0]   - base['R2'][0]
        dRMSE = metric_summary['RMSE'][0] - base['RMSE'][0]
        dMAE  = metric_summary['MAE'][0]  - base['MAE'][0]
        print(f"  Δ ({label}):  R²={dR2:+.4f}   MAE={dMAE:+.3f}   RMSE={dRMSE:+.3f}")

    print()
    delta(fusion_summary, reference_summary, "Fusion − Reference")
    delta(fedavg_summary, reference_summary, "FedAvg − Reference")
    delta(fusion_summary, fedavg_summary,  "Fusion − FedAvg")

    def ashrae_check(m, label):
        cv = m['CV_RMSE_pct']; nmbe = abs(m['NMBE_pct'])
        print(f"  {label:58s}  CV(RMSE)={cv:6.2f}%  ({'✅' if cv <= 30 else '❌'})  "
              f"|NMBE|={nmbe:6.2f}%  ({'✅' if nmbe <= 10 else '❌'})")

    print("\n" + "=" * 118)
    print("ASHRAE Guideline 14 compliance (hourly: CV(RMSE) ≤ 30%, |NMBE| ≤ 10%)")
    print("=" * 118)
    for i, (client_data, cm, fm, am) in enumerate(
            zip(client_datasets, reference_metrics, fusion_metrics, fedavg_metrics), 1):
        ashrae_check(cm, f"C{i} ({client_data['name']}) — Reference")
        ashrae_check(fm, f"C{i} ({client_data['name']}) — Fusion")
        ashrae_check(am, f"C{i} ({client_data['name']}) — FedAvg")
    print("  " + "-" * 112)
    ashrae_check({"CV_RMSE_pct": reference_summary["CV_RMSE_pct"][0],
                  "NMBE_pct":    reference_summary["NMBE_pct"][0]},
                 "MEAN Reference")
    ashrae_check({"CV_RMSE_pct": fusion_summary["CV_RMSE_pct"][0],
                  "NMBE_pct":    fusion_summary["NMBE_pct"][0]},
                 "MEAN Hetero-FL fusion")
    ashrae_check({"CV_RMSE_pct": fedavg_summary["CV_RMSE_pct"][0],
                  "NMBE_pct":    fedavg_summary["NMBE_pct"][0]},
                 "MEAN FedAvg")


if __name__ == "__main__":
    main()
