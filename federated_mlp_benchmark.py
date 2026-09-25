# -*- coding: utf-8 -*-
"""Federated submodel benchmark for UCI Letter Recognition.

Compares StitchFL, HaloFL, FedGSE, SubFLOT, and full-model FedAvg under
a shared federated training budget. The dataset is partitioned into balanced
stratified clients and evaluated with multiclass classification metrics.
"""

import argparse
import math
import pathlib
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import linprog, minimize
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

BASE_DIR = pathlib.Path.cwd()

DATA_DIR_CANDIDATES = [
    BASE_DIR / "Datasets" / "letter_recognition",
    BASE_DIR / "Datasets" / "letter+recognition",
]

LETTER_FEATURE_NAMES = [
    "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
    "y2bar", "xybar", "x2ybar", "xy2bar", "x-ege", "xegvx", "y-ege", "yegvx",
]
LABEL_COLUMN = "letter"

RESULTS_DIR = BASE_DIR / "results" / "FL" / "MLP" / "letter_recognition"

RESULTS_CSV = {
    "stitchfl": RESULTS_DIR / "stitchfl.csv",
    "halofl":   RESULTS_DIR / "halofl.csv",
    "fedgse":   RESULTS_DIR / "fedgse.csv",
    "subflot":  RESULTS_DIR / "subflot.csv",
    "fedavg":   RESULTS_DIR / "fedavg.csv",
}

GLOBAL_HIDDEN_SIZES = [128, 64, 32]
LEARNING_RATE = 0.005

METRIC_AVERAGE = "macro"

FL_ROUNDS = 10
FL_LOCAL_EPOCHS = 5
FL_BATCH_SIZE = 32

FL_VERBOSE_ROUND_EVAL = True

CLIENT_RANGES = [
    [(0, 44), (0, 12), (0, 8)],
    [(12, 72), (10, 27), (4, 15)],
    [(40, 84), (25, 37), (11, 19)],
    [(52, 104), (36, 50), (16, 25)],
    [(73, 128), (49, 64), (22, 32)],
]

NUM_CLIENTS = len(CLIENT_RANGES)
NUM_HIDDEN_LAYERS = len(GLOBAL_HIDDEN_SIZES)

CLIENT_WIDTHS = [[e - s for (s, e) in r] for r in CLIENT_RANGES]

TEST_FRACTION = 0.2

STITCH_ROUNDS = FL_ROUNDS
STITCH_EPOCHS_PER_CLIENT = FL_LOCAL_EPOCHS

STITCH_RECALIBRATE = True
STITCH_DISTILL_SCOPE = "all"
STITCH_N_ANCHOR = 20000
STITCH_CLIP_Z = 3.0
STITCH_DISTILL_EPOCHS = 20
STITCH_DISTILL_LR = 0.002
STITCH_DISTILL_BATCH = 128
STITCH_CE_L2 = 1e-4
STITCH_CE_MAXITER = 300

STITCH_EVAL_ENSEMBLE = True

STITCH_MODE = "factor"
STITCH_TOPK = None

STITCH_ZERO_UNCOVERED = True

HALO_ROUNDS = FL_ROUNDS
HALO_EPOCHS_PER_CLIENT = FL_LOCAL_EPOCHS
HALO_BUDGET_MODE = "per_layer"
HALO_ALPHA_UIL = 0.5
HALO_ACT_THETA = 0.0
HALO_P_MIN = 0.0
HALO_P_MAX = 0.85

HALO_LPB_ENABLE = True

HALO_AGG_MODE = "keep"

HALO_DIVERSIFY = True

FEDGSE_ROUNDS = FL_ROUNDS
FEDGSE_LOCAL_EPOCHS = FL_LOCAL_EPOCHS
FEDGSE_BATCH_SIZE = FL_BATCH_SIZE

FEDGSE_CAPACITIES = [
    float(np.mean([w / h for w, h in zip(widths, GLOBAL_HIDDEN_SIZES)]))
    for widths in CLIENT_WIDTHS
]

FEDGSE_SYNTH_GENERATOR = "gaussian"
FEDGSE_SYNTH_PER_CLIENT = 260

FEDGSE_GMM_COMPONENTS = 3
FEDGSE_SYNTH_COV_REG = 1e-4

FEDGSE_MAX_SIMILAR = 512
FEDGSE_MIN_SIMILAR = 8

SUBFLOT_GLOBAL_ROUNDS = FL_ROUNDS
SUBFLOT_LOCAL_EPOCHS = FL_LOCAL_EPOCHS
SUBFLOT_BATCH_SIZE = FL_BATCH_SIZE
SUBFLOT_MIX_RATIO = 0.5

SUBFLOT_SAR_LAMBDA = 0.01
SUBFLOT_DYNAMIC_MU = True
SUBFLOT_WARMUP_ROUNDS = 1
SUBFLOT_OT_SOLVER = "emd"
SUBFLOT_SINKHORN_REG = 0.01

SUBFLOT_SKIP_HEAD = False

FEDAVG_ROUNDS = FL_ROUNDS
FEDAVG_LOCAL_EPOCHS = FL_LOCAL_EPOCHS
FEDAVG_BATCH_SIZE = FL_BATCH_SIZE

def reseed_random():

    seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def rand_seed():

    return int(np.random.SeedSequence().entropy % (2 ** 31))

def _find_letter_file():
    preferred_names = (
        "letter-recognition.data",
        "letter-recognition.csv",
        "letter_recognition.data",
        "letter.data",
    )
    for data_dir in DATA_DIR_CANDIDATES:
        for filename in preferred_names:
            path = data_dir / filename
            if path.exists():
                return path
        if data_dir.is_dir():
            for pattern in ("*.data", "*.csv", "*.txt"):
                files = sorted(
                    path for path in data_dir.glob(pattern)
                    if "names" not in path.name.lower()
                )
                if files:
                    return files[0]
    searched = ", ".join(str(path) for path in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(
        f"Letter Recognition data file not found. Searched: {searched}"
    )

def _read_letter_dataframe(path):

    df = pd.read_csv(path, header=None, sep=",", skipinitialspace=True)

    def _is_label_col(col):
        vals = col.astype(str).str.strip()
        return vals.str.fullmatch(r"[A-Za-z]").fillna(False).mean() > 0.9

    first_row = df.iloc[0].astype(str).str.strip()
    numeric_first_row = pd.to_numeric(first_row, errors="coerce").notna()
    if numeric_first_row.sum() < df.shape[1] - 1:
        df = df.iloc[1:].reset_index(drop=True)

    if not _is_label_col(df.iloc[:, 0]) and not _is_label_col(df.iloc[:, -1]):
        raise ValueError(
            f"Could not identify the label column in {path}."
        )

    if _is_label_col(df.iloc[:, 0]):
        labels = df.iloc[:, 0]
        features = df.iloc[:, 1:]
    else:
        labels = df.iloc[:, -1]
        features = df.iloc[:, :-1]

    if features.shape[1] != len(LETTER_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(LETTER_FEATURE_NAMES)} features y se han "
            f"encontrado {features.shape[1]} en {path}."
        )

    out = features.copy()
    out.columns = LETTER_FEATURE_NAMES
    out = out.apply(pd.to_numeric, errors="raise")
    out.insert(0, LABEL_COLUMN, labels.astype(str).str.strip().str.upper().values)
    return out.dropna().reset_index(drop=True)

def load_data_and_clients():
    """Load, stratify, scale, and partition Letter Recognition across clients."""

    data_path = _find_letter_file()
    df = _read_letter_dataframe(data_path)
    print(f"Dataset: {data_path} | {len(df)} samples "
          f"| {len(LETTER_FEATURE_NAMES)} features")

    x_all = df.loc[:, LETTER_FEATURE_NAMES]
    raw_labels = df.loc[:, LABEL_COLUMN]

    encoder = LabelEncoder().fit(sorted(raw_labels.unique()))
    class_names = list(encoder.classes_)
    n_classes = len(class_names)
    label_indices = encoder.transform(raw_labels)
    eye = np.eye(n_classes, dtype=np.float32)
    print(f"Detected classes ({n_classes}): {''.join(map(str, class_names))}")

    skf = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True,
                          random_state=rand_seed())
    partitions = [part_idx for _, part_idx in skf.split(x_all, label_indices)]

    raw_clients = []
    for client_id, part_idx in enumerate(partitions):
        X_client = x_all.iloc[part_idx]
        y_client = label_indices[part_idx]
        x_train, x_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=TEST_FRACTION, stratify=y_client
        )
        raw_clients.append((x_train, x_test, y_train, y_test))
        print(f"Client {client_id}: {len(part_idx)} samples "
              f"({len(x_train)} train / {len(x_test)} test) "
              f"| classes_present={len(np.unique(y_client))}")

    x_train_all = pd.concat([rc[0] for rc in raw_clients], axis=0)
    scaler = StandardScaler()
    scaler.fit(x_train_all)

    clients = []
    for (x_train, x_test, y_train, y_test) in raw_clients:
        clients.append({
            "X_tr": scaler.transform(x_train).astype(np.float32),
            "X_te": scaler.transform(x_test).astype(np.float32),
            "y_tr": eye[y_train],
            "y_te": eye[y_test],
            "y_tr_idx": np.asarray(y_train, dtype=np.int64),
            "y_te_idx": np.asarray(y_test, dtype=np.int64),
        })

    return {
        "feature_columns": LETTER_FEATURE_NAMES,
        "target_columns": class_names,
        "target_names": class_names,
        "class_names": class_names,
        "input_dim": len(LETTER_FEATURE_NAMES),
        "n_targets": n_classes,
        "n_classes": n_classes,
        "scaler": scaler,
        "encoder": encoder,
        "clients": clients,
    }

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "MCC"]

def _to_class_indices(y):

    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(int)
    if y.shape[1] == 1:
        return y.ravel().astype(int)
    return np.argmax(y, axis=1).astype(int)

def _classification_metrics(y_true, y_pred, n_classes):

    yt = _to_class_indices(y_true)
    yp = _to_class_indices(y_pred)
    labels = np.arange(n_classes)
    kw = dict(labels=labels, average=METRIC_AVERAGE, zero_division=0)
    return {
        "Accuracy": accuracy_score(yt, yp),
        "Precision": precision_score(yt, yp, **kw),
        "Recall": recall_score(yt, yp, **kw),
        "F1": f1_score(yt, yp, **kw),
        "MCC": matthews_corrcoef(yt, yp),
    }

def evaluate_over_clients(predict_fn, clients, target_names, title):
    """Evaluate a global predictor on each client test set and average metrics."""

    n_classes = len(target_names)
    per_client = []
    for cid, cl in enumerate(clients):
        y_pred = np.asarray(predict_fn(cl["X_te"]))
        m = _classification_metrics(cl["y_te"], y_pred, n_classes)
        per_client.append({"Output": f"Client_{cid}", **m})

    client_metrics_df = pd.DataFrame(per_client, columns=["Output"] + METRIC_COLUMNS)
    mean_between_clients = client_metrics_df[METRIC_COLUMNS].mean(axis=0, skipna=True)
    average_row = {"Output": "Average", **mean_between_clients.to_dict()}
    results_df = pd.concat([client_metrics_df, pd.DataFrame([average_row])], ignore_index=True)

    print(f"\n=== {title} ===")
    print(tabulate(results_df, headers="keys", tablefmt="github",
                   floatfmt=".4f", showindex=True))
    return results_df

def round_monitor_accuracy(predict_fn, clients):

    accs = []
    for cl in clients:
        y_pred = np.asarray(predict_fn(cl["X_te"]))
        accs.append(accuracy_score(_to_class_indices(cl["y_te"]),
                                   _to_class_indices(y_pred)))
    return float(np.mean(accs))

def print_round_progress(method, round_id, total_rounds, predict_fn, clients):

    if not FL_VERBOSE_ROUND_EVAL:
        print(f"  [{method}] Round {round_id}/{total_rounds} | aggregation applied "
              f"on the server")
        return
    acc = round_monitor_accuracy(predict_fn, clients)
    print(f"  [{method}] Round {round_id}/{total_rounds} | aggregation applied "
          f"| Mean client accuracy = {acc:.4f}")

def save_average_row(results_df, method_key):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_CSV[method_key]
    average = results_df.loc[results_df["Output"] == "Average"].copy()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    average.to_csv(csv_path, mode="a", header=write_header, index=False)
    print(f"\n[{method_key}] Average row appended to: {csv_path}")

def build_server_model_keras(input_dim, n_targets):
    m = Sequential([Input(shape=(input_dim,))])
    for h in GLOBAL_HIDDEN_SIZES:
        m.add(Dense(h, activation="relu"))
    m.add(Dense(n_targets, activation="softmax"))
    m.compile(optimizer=Adam(LEARNING_RATE), loss="categorical_crossentropy")
    return m

def get_dense_layers(model):
    return [lyr for lyr in model.layers if isinstance(lyr, Dense)]

def _logit_fn(model):

    hidden = Model(model.inputs, model.layers[-2].output)
    W, b = model.layers[-1].get_weights()
    W64, b64 = np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return lambda X: hidden.predict(X, verbose=0).astype(np.float64) @ W64 + b64

def ensemble_logit_predict(client_models):

    fs = [_logit_fn(m) for m in client_models]
    return lambda X: np.mean([f(X) for f in fs], axis=0)

def teacher_probs(client_models, X_anchor):

    Z = np.mean([_logit_fn(c)(X_anchor) for c in client_models], axis=0)
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)

def covered_mask(input_dim, n_targets):

    dims = [input_dim] + GLOBAL_HIDDEN_SIZES + [n_targets]
    masks = []
    for l in range(NUM_HIDDEN_LAYERS + 1):
        cov = np.zeros((dims[l], dims[l + 1]), dtype=bool)
        for ranges in CLIENT_RANGES:
            rows = np.arange(input_dim) if l == 0 else np.arange(*ranges[l - 1])
            cols = np.arange(n_targets) if l == NUM_HIDDEN_LAYERS else np.arange(*ranges[l])
            if len(rows) and len(cols):
                cov[np.ix_(rows, cols)] = True
        masks.append(cov)
    return masks

def _fit_head_ce(H, P):

    H = np.asarray(H, np.float64); P = np.asarray(P, np.float64)
    A = np.hstack([H, np.ones((H.shape[0], 1))])
    n, d = A.shape; k = P.shape[1]
    reg = np.ones((d, k)); reg[-1, :] = 0.0

    def fun(w):
        Wb = w.reshape(d, k)
        Z = A @ Wb; Z -= Z.max(axis=1, keepdims=True)
        ls = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        loss = -(P * ls).sum() / n + 0.5 * STITCH_CE_L2 * ((Wb * reg) ** 2).sum()
        grad = A.T @ (np.exp(ls) - P) / n + STITCH_CE_L2 * Wb * reg
        return loss, grad.ravel()

    r = minimize(fun, np.zeros(d * k), jac=True, method="L-BFGS-B",
                 options={"maxiter": STITCH_CE_MAXITER})
    Wb = r.x.reshape(d, k)
    return Wb[:-1, :].astype(np.float32), Wb[-1, :].astype(np.float32)

def distill_full(server_model, X_anchor, P_teacher, masks):

    server_model.compile(optimizer=Adam(STITCH_DISTILL_LR),
                         loss="categorical_crossentropy")
    dense = get_dense_layers(server_model)
    for _ in range(STITCH_DISTILL_EPOCHS):
        server_model.fit(X_anchor, P_teacher, epochs=1,
                         batch_size=STITCH_DISTILL_BATCH, verbose=0)
        for lyr, m in zip(dense, masks):
            W, b = lyr.get_weights()
            lyr.set_weights([np.where(m, W, 0.0), b])

def run_stitchfl(data):
    """Run the StitchFL federated training and fusion pipeline."""
    print("\n" + "#" * 78)
    print("# METHOD 1/4: StitchFL (range-based partitioning + analytical stitching)")
    print("#" * 78)
    reseed_random()
    input_dim, n_targets = data["input_dim"], data["n_targets"]
    clients = data["clients"]
    target_names = data["target_names"]

    def build_client_model_dynamic(layer_ranges):
        m = Sequential([Input(shape=(input_dim,))])
        for (s, e) in layer_ranges:
            if e > s:
                m.add(Dense(e - s, activation="relu"))
        m.add(Dense(n_targets, activation="softmax"))
        m.compile(optimizer=Adam(LEARNING_RATE), loss="categorical_crossentropy")
        return m

    def present_node_ranges(layer_ranges):
        nodes = [(0, (0, input_dim))]
        nodes += [(i, (s_, e_)) for i, (s_, e_) in enumerate(layer_ranges, start=1)
                  if e_ > s_]
        nodes.append((NUM_HIDDEN_LAYERS + 1, (0, n_targets)))
        return nodes

    def copy_consecutive_blocks(model, layer_ranges, server):

        nodes = present_node_ranges(layer_ranges)
        s_dense = get_dense_layers(server)
        c_dense = get_dense_layers(model)
        for k, ((u, (su, eu)), (v, (sv, ev))) in enumerate(zip(nodes[:-1], nodes[1:])):
            if v != u + 1:
                continue
            W, b = s_dense[u].get_weights()
            rows = np.arange(input_dim) if u == 0 else np.arange(su, eu)
            cols = np.arange(n_targets) if v == NUM_HIDDEN_LAYERS + 1 else np.arange(sv, ev)
            W_cur, _ = c_dense[k].get_weights()
            W_slice = W[np.ix_(rows, cols)]
            if W_slice.shape == W_cur.shape:
                c_dense[k].set_weights([W_slice.copy(), b[cols].copy()])

    def node_idx(node_id, layer_ranges, dims):
        if node_id == 0:
            return np.arange(dims[0])
        if node_id == NUM_HIDDEN_LAYERS + 1:
            return None
        s, e = layer_ranges[node_id - 1]
        return np.arange(s, e)

    def present_nodes(layer_ranges):
        hs = [i for i, (s, e) in enumerate(layer_ranges, start=1) if e > s]
        return [0] + hs + [NUM_HIDDEN_LAYERS + 1]

    def dense_layers_of(model):
        return [lyr for lyr in model.layers if isinstance(lyr, Dense)]

    def safe_div(num, den):
        return num / np.where(den == 0, 1, den)

    def client_edges(model, layer_ranges, dims):
        pn = present_nodes(layer_ranges)
        dense = dense_layers_of(model)
        assert len(dense) == (len(pn) - 1)
        edges = []
        for k, (u, v) in enumerate(zip(pn[:-1], pn[1:])):
            W, b = dense[k].get_weights()
            edges.append({
                "u": u, "v": v,
                "from_idx": node_idx(u, layer_ranges, dims),
                "to_idx": node_idx(v, layer_ranges, dims),
                "W": W, "b": b,
            })
        return edges

    def get_free_neurons(k, client_ranges):

        used_neurons = np.zeros(GLOBAL_HIDDEN_SIZES[k - 1], dtype=bool)
        for r in client_ranges:
            s_, e_ = r[k - 1]
            used_neurons[s_:e_] = True
        return np.where(~used_neurons)[0]

    def progressive_fuse_dynamic(client_models, server, client_ranges):
        dims = [input_dim] + GLOBAL_HIDDEN_SIZES + [n_targets]
        W_sum = [np.zeros((dims[u], dims[u + 1])) for u in range(NUM_HIDDEN_LAYERS + 1)]
        W_cnt = [np.zeros_like(W_sum[u]) for u in range(NUM_HIDDEN_LAYERS + 1)]
        b_sum = [np.zeros(dims[u + 1]) for u in range(NUM_HIDDEN_LAYERS + 1)]
        b_cnt = [np.zeros_like(b_sum[u]) for u in range(NUM_HIDDEN_LAYERS + 1)]
        pending = []
        n_factor = 0

        fixed_weight_mask = [np.zeros_like(W_sum[u], dtype=bool) for u in range(NUM_HIDDEN_LAYERS + 1)]
        fixed_bias_mask = [np.zeros_like(b_sum[u], dtype=bool) for u in range(NUM_HIDDEN_LAYERS + 1)]

        def insert_consecutive(edge):
            u, v = edge["u"], edge["v"]
            assert v == u + 1
            rows = edge["from_idx"]
            cols = np.arange(n_targets) if v == NUM_HIDDEN_LAYERS + 1 else edge["to_idx"]
            W_sum[u][np.ix_(rows, cols)] += edge["W"]
            W_cnt[u][np.ix_(rows, cols)] += 1
            b_sum[u][cols] += edge["b"]
            b_cnt[u][cols] += 1

        def insert_factorized(edge):

            u, v = edge["u"], edge["v"]
            if v != u + 2 or u < 1:
                return False
            k = u + 1
            free_neurons = get_free_neurons(k, client_ranges)
            d = len(edge["from_idx"])
            if len(free_neurons) < d:
                return False
            cols = free_neurons[:d]
            W1 = np.zeros((d, d)); np.fill_diagonal(W1, 1.0)
            insert_consecutive({"u": u, "v": k, "from_idx": edge["from_idx"],
                                "to_idx": cols, "W": W1, "b": np.zeros(d)})
            insert_consecutive({"u": k, "v": v, "from_idx": cols,
                                "to_idx": edge["to_idx"],
                                "W": edge["W"], "b": edge["b"]})
            fixed_weight_mask[u][np.ix_(edge["from_idx"], cols)] = True
            fixed_weight_mask[k][np.ix_(cols, edge["to_idx"])] = True
            fixed_bias_mask[u][cols] = True
            fixed_bias_mask[k][edge["to_idx"]] = True
            return True

        def try_push_one_step(edge):
            u, v = edge["u"], edge["v"]
            if v == u + 1:
                insert_consecutive(edge)
                return None
            if u > NUM_HIDDEN_LAYERS:
                return edge
            rows_from = edge["from_idx"]
            row_support = W_cnt[u][rows_from, :].sum(axis=1)
            row_mask = row_support > 0
            if not np.any(row_mask):
                return edge
            rows_use = rows_from[row_mask]
            W_A = edge["W"][row_mask, :]
            col_support = W_cnt[u][rows_use, :].sum(axis=0)
            cols = np.where(col_support > 0)[0]
            if cols.size == 0:
                return edge
            if STITCH_TOPK is not None and cols.size > STITCH_TOPK:
                scores = col_support[cols]
                cols = cols[np.argsort(scores)[-STITCH_TOPK:]]
            den = W_cnt[u][np.ix_(rows_use, cols)]
            W_B = W_sum[u][np.ix_(rows_use, cols)] / np.where(den == 0, 1, den)
            bden = b_cnt[u][cols]
            b_B = b_sum[u][cols] / np.where(bden == 0, 1, bden)
            W_new = np.linalg.pinv(W_B) @ W_A
            b_new = edge["b"] - (b_B @ W_new)
            return {"u": u + 1, "v": v, "from_idx": cols,
                    "to_idx": edge["to_idx"], "W": W_new, "b": b_new}

        def process_pending():
            nonlocal pending
            for _ in range(NUM_HIDDEN_LAYERS + 2):
                if not pending:
                    return
                progressed = False
                new_pending = []
                for e in pending:
                    out = try_push_one_step(e)
                    if out is None:
                        progressed = True
                    else:
                        progressed = progressed or (out["u"] != e["u"])
                        new_pending.append(out)
                pending = new_pending
                if not progressed:
                    return

        for i, (c, r) in enumerate(zip(client_models, client_ranges), 1):
            for e in client_edges(c, r, dims):
                if e["v"] == e["u"] + 1:
                    insert_consecutive(e)
                elif STITCH_MODE == "factor" and insert_factorized(e):
                    n_factor += 1
                elif STITCH_MODE == "off":
                    pass
                else:
                    pending.append(e)
            if STITCH_MODE == "pinv":
                process_pending()

        for u in range(NUM_HIDDEN_LAYERS + 1):
            W_old, b_old = server.layers[u].get_weights()
            W_mean = safe_div(W_sum[u], W_cnt[u])
            b_mean = safe_div(b_sum[u], b_cnt[u])
            W_avg = np.where(W_cnt[u] > 0, W_mean, W_old)
            b_avg = np.where(b_cnt[u] > 0, b_mean, b_old)

            W_avg = np.where(fixed_weight_mask[u], W_mean, W_avg)
            b_avg = np.where(fixed_bias_mask[u], b_mean, b_avg)
            server.layers[u].set_weights([W_avg, b_avg])
        return len(pending), n_factor

    server_model = build_server_model_keras(input_dim, n_targets)

    if STITCH_ZERO_UNCOVERED:
        dims = [input_dim] + GLOBAL_HIDDEN_SIZES + [n_targets]
        s_dense = get_dense_layers(server_model)
        n_zeroed = n_total = 0
        for l in range(NUM_HIDDEN_LAYERS + 1):
            covered = np.zeros((dims[l], dims[l + 1]), dtype=bool)
            for ranges in CLIENT_RANGES:
                rows = np.arange(input_dim) if l == 0 else np.arange(*ranges[l - 1])
                cols = np.arange(n_targets) if l == NUM_HIDDEN_LAYERS else np.arange(*ranges[l])
                covered[np.ix_(rows, cols)] = True
            W, b = s_dense[l].get_weights()
            s_dense[l].set_weights([np.where(covered, W, 0.0), b])
            n_zeroed += int((~covered).sum())
            n_total += covered.size
        print(f"  [FIX-STITCH-2] {n_zeroed}/{n_total} celdas "
              f"({n_zeroed / n_total:.1%}) are uncovered -> set to zero")

    client_models = [build_client_model_dynamic(r) for r in CLIENT_RANGES]
    n_pending = n_factor = 0
    for rnd in range(1, STITCH_ROUNDS + 1):
        print(f"\n  {'=' * 12} ROUND STITCHFL {rnd}/{STITCH_ROUNDS} {'=' * 12}")
        for i, (cl, ranges, c) in enumerate(
                zip(clients, CLIENT_RANGES, client_models), 1):

            copy_consecutive_blocks(c, ranges, server_model)

            c.fit(cl["X_tr"], cl["y_tr"], epochs=STITCH_EPOCHS_PER_CLIENT,
                  batch_size=FL_BATCH_SIZE, verbose=0)
            if rnd == 1:
                print(f"    client {i} | ranges={ranges} "
                      f"| units={[e - s for (s, e) in ranges]}")

        n_pending, n_factor = progressive_fuse_dynamic(
            client_models, server_model, CLIENT_RANGES)
        print_round_progress("StitchFL", rnd, STITCH_ROUNDS,
                             lambda X: server_model.predict(X, verbose=0), clients)
    if n_factor:
        print(f"  [stitch] {n_factor} skip-edge(s) encaminados por construccion "
              f"analitica (modo '{STITCH_MODE}')")
    if n_pending:
        print(f"  [stitch] WARNING: {n_pending} unstitched skip edge(s). With mode "
              f"'factor', the skipped layer has insufficient free neurons; "
              f"increase GLOBAL_HIDDEN_SIZES for that layer.")

    if STITCH_RECALIBRATE:
        print(f"\n  [recal] Gaussian anchors x{STITCH_N_ANCHOR} | "
              f"scope '{STITCH_DISTILL_SCOPE}'")
        rs_a = np.random.default_rng()
        X_anchor = np.clip(rs_a.normal(size=(STITCH_N_ANCHOR, input_dim)),
                           -STITCH_CLIP_Z, STITCH_CLIP_Z).astype(np.float32)
        P_teacher = teacher_probs(client_models, X_anchor)

        acc_antes = round_monitor_accuracy(
            lambda X: server_model.predict(X, verbose=0), clients)

        if STITCH_DISTILL_SCOPE == "all":
            distill_full(server_model, X_anchor, P_teacher.astype(np.float32),
                         covered_mask(input_dim, n_targets))
        else:
            hidden_model = Model(server_model.inputs, server_model.layers[-2].output)
            H = hidden_model.predict(X_anchor, verbose=0)
            W_new, b_new = _fit_head_ce(H, P_teacher)
            server_model.layers[-1].set_weights([W_new, b_new])

        acc_despues = round_monitor_accuracy(
            lambda X: server_model.predict(X, verbose=0), clients)
        print(f"  [recal] Mean client accuracy: {acc_antes:.4f} -> "
              f"{acc_despues:.4f} ({acc_despues - acc_antes:+.4f})")
        if acc_despues < acc_antes:
            print("  [recal] WARNING: distillation reduces performance for this partition. "
                  "It is kept to avoid cherry-picking; use --no-recalibration "
                  "to disable it.")

    results_df = evaluate_over_clients(
        lambda X: server_model.predict(X, verbose=0), clients, target_names,
        "StitchFL: final fused global model (mean across clients)",
    )
    save_average_row(results_df, "stitchfl")

    if STITCH_EVAL_ENSEMBLE:
        pred_ens = ensemble_logit_predict(client_models)
        df_ens = evaluate_over_clients(
            pred_ens, clients, target_names,
            "ens_logit_clientes: submodel ensemble by mean logits "
            "(REFERENCE, not a single model)")
        save_average_row(df_ens, "stitchfl_ens_logit")
        acc_g = results_df.loc[results_df["Output"] == "Average", "Accuracy"].iloc[0]
        acc_e = df_ens.loc[df_ens["Output"] == "Average", "Accuracy"].iloc[0]
        print(f"\n  [ens] fused model {acc_g:.4f} | ensemble {acc_e:.4f} "
              f"({acc_g - acc_e:+.4f})")
        print("  [ens] The ensemble requires all five submodels at inference; "
              "StitchFL produces one model. It is a reference upper bound, not a competitor.")
        EXTRA_RESULTS["ens_logit_clientes"] = df_ens

    return results_df

def run_halofl(data):
    """Run the HaloFL federated training and positional aggregation pipeline."""
    print("\n" + "#" * 78)
    print("# METHOD 2/4: HaloFL (UIL mask extraction + LPB + positional aggregation)")
    print("#" * 78)
    reseed_random()
    input_dim, n_targets = data["input_dim"], data["n_targets"]
    clients = data["clients"]
    target_names = data["target_names"]

    client_units = [[e - s for (s, e) in r] for r in CLIENT_RANGES]
    client_sparsity_per_layer = [
        [1.0 - u / h for u, h in zip(units, GLOBAL_HIDDEN_SIZES)] for units in client_units
    ]
    client_sparsity = [float(np.mean(ps)) for ps in client_sparsity_per_layer]

    def build_submodel(hidden_sizes):
        m = Sequential([Input(shape=(input_dim,))])
        for h in hidden_sizes:
            m.add(Dense(int(h), activation="relu"))
        m.add(Dense(n_targets, activation="softmax"))
        m.compile(optimizer=Adam(LEARNING_RATE), loss="categorical_crossentropy")
        return m

    def retained_units(sparsity, layer_size):
        p = min(max(sparsity, HALO_P_MIN), HALO_P_MAX)
        k = int(round((1.0 - p) * layer_size))
        return max(1, min(layer_size, k))

    def lpb_adjust(retained, layer_act_ratio):

        if not HALO_LPB_ENABLE or NUM_HIDDEN_LAYERS < 2:
            return list(retained)
        R = np.clip(np.asarray(layer_act_ratio, dtype=float), 1e-6, None)
        order = np.argsort(R)
        new = list(retained)
        for k in range(NUM_HIDDEN_LAYERS // 2):
            a, b = int(order[k]), int(order[NUM_HIDDEN_LAYERS - 1 - k])
            if a == b:
                continue

            delta = retained[a] * (np.log(R[b]) - np.log(R[a])) / R[b]
            delta = int(np.floor(max(0.0, delta)))

            floor_a = max(1, retained[a] // 2)
            delta = max(0, min(delta, new[a] - floor_a, GLOBAL_HIDDEN_SIZES[b] - new[b]))
            new[a] -= delta
            new[b] += delta
        return new

    def compute_local_uil(submodel, x_client):

        hidden_layers = [lyr for lyr in submodel.layers if isinstance(lyr, Dense)][:-1]
        extractor = Model(submodel.inputs, [h.output for h in hidden_layers])
        acts = extractor.predict(x_client, verbose=0)
        if NUM_HIDDEN_LAYERS == 1:
            acts = [acts]
        luil_per_layer, layer_act_ratio = [], []
        for A in acts:
            A = np.asarray(A)
            mask = A > HALO_ACT_THETA
            R = mask.mean(axis=0)
            S = (A * mask).sum(axis=0)
            S_n = S / (S.max() if S.max() > 0 else 1.0)
            R_n = R / (R.max() if R.max() > 0 else 1.0)
            luil_per_layer.append(HALO_ALPHA_UIL * S_n + (1.0 - HALO_ALPHA_UIL) * R_n)
            layer_act_ratio.append(float(R.mean()))
        return luil_per_layer, layer_act_ratio

    def extract_submodel_weights(server_model, sel_idx):
        dense = get_dense_layers(server_model)
        weights = []
        prev_idx = np.arange(input_dim)
        for l, lyr in enumerate(dense):
            W, b = lyr.get_weights()
            if l < NUM_HIDDEN_LAYERS:
                cols = sel_idx[l]
                weights.append([W[np.ix_(prev_idx, cols)], b[cols]])
                prev_idx = cols
            else:
                weights.append([W[prev_idx, :], b])
        return weights

    def select_units(uil, k_per_layer, client_idx=0):

        sel = []
        for l in range(NUM_HIDDEN_LAYERS):
            k = k_per_layer[l]
            order = np.argsort(uil[l])[::-1]
            if HALO_DIVERSIFY:
                off = (client_idx * k) % max(1, GLOBAL_HIDDEN_SIZES[l] - k + 1)
                idx = order[off:off + k]
            else:
                idx = order[:k]
            sel.append(np.sort(idx))
        return sel

    def init_uil_from_norms(server_model):

        dense = get_dense_layers(server_model)
        uil = []
        for l in range(NUM_HIDDEN_LAYERS):
            W = dense[l].get_weights()[0]
            n = np.linalg.norm(W, axis=0)
            uil.append(n / (n.max() if n.max() > 0 else 1.0))
        return uil

    def halofl_round(server_model, uil_global, round_id):
        dense = get_dense_layers(server_model)
        W_sum = [np.zeros_like(lyr.get_weights()[0]) for lyr in dense]
        W_cnt = [np.zeros_like(lyr.get_weights()[0]) for lyr in dense]
        b_sum = [np.zeros_like(lyr.get_weights()[1]) for lyr in dense]
        b_cnt = [np.zeros_like(lyr.get_weights()[1]) for lyr in dense]

        luil_sum = [np.zeros(h) for h in GLOBAL_HIDDEN_SIZES]
        luil_cnt = [np.zeros(h) for h in GLOBAL_HIDDEN_SIZES]

        round_ratios = []

        for i, cl in enumerate(clients, 1):
            x_client, y_client = cl["X_tr"], cl["y_tr"]
            sparsity = client_sparsity[i - 1]

            if HALO_BUDGET_MODE == "per_layer":
                base = [max(1, min(GLOBAL_HIDDEN_SIZES[l], client_units[i - 1][l]))
                        for l in range(NUM_HIDDEN_LAYERS)]
            else:
                base = [retained_units(sparsity, GLOBAL_HIDDEN_SIZES[l]) for l in range(NUM_HIDDEN_LAYERS)]

            if round_id == 1 or not HALO_LPB_ENABLE:
                k_per_layer = list(base)
            else:
                k_per_layer = lpb_adjust(base, server_model._halo_layer_ratio)

            sel_idx = select_units(uil_global, k_per_layer, client_idx=i - 1)

            sub_w = extract_submodel_weights(server_model, sel_idx)
            sub = build_submodel([len(s) for s in sel_idx])
            sub.set_weights([wb_part for wb in sub_w for wb_part in wb])
            sub.fit(x_client, y_client, epochs=HALO_EPOCHS_PER_CLIENT,
                    batch_size=FL_BATCH_SIZE, verbose=0)

            luil, act_ratio = compute_local_uil(sub, x_client)
            round_ratios.append(act_ratio)

            sub_dense = get_dense_layers(sub)
            prev_idx = np.arange(input_dim)
            for l, lyr in enumerate(sub_dense):
                Wl, bl = lyr.get_weights()
                if l < NUM_HIDDEN_LAYERS:
                    cols = sel_idx[l]
                    W_sum[l][np.ix_(prev_idx, cols)] += Wl
                    W_cnt[l][np.ix_(prev_idx, cols)] += 1
                    b_sum[l][cols] += bl
                    b_cnt[l][cols] += 1
                    prev_idx = cols
                else:
                    W_sum[l][prev_idx, :] += Wl
                    W_cnt[l][prev_idx, :] += 1
                    b_sum[l] += bl
                    b_cnt[l] += 1

            for l in range(NUM_HIDDEN_LAYERS):
                luil_sum[l][sel_idx[l]] += luil[l]
                luil_cnt[l][sel_idx[l]] += 1

            print(f"  [R{round_id}] client {i} | p={sparsity:.2f} "
                  f"| units={[len(s) for s in sel_idx]}")

        N = len(clients)
        new_weights = []
        for l, lyr in enumerate(dense):
            if HALO_AGG_MODE == "paper":

                W_avg = W_sum[l] / N
                b_avg = b_sum[l] / N
            elif HALO_AGG_MODE == "keep":

                W_old, b_old = lyr.get_weights()
                W_avg = np.where(W_cnt[l] > 0,
                                 W_sum[l] / np.maximum(W_cnt[l], 1), W_old)
                b_avg = np.where(b_cnt[l] > 0,
                                 b_sum[l] / np.maximum(b_cnt[l], 1), b_old)
            else:

                W_avg = np.where(W_cnt[l] > 0,
                                 W_sum[l] / np.maximum(W_cnt[l], 1), 0.0)
                b_avg = np.where(b_cnt[l] > 0,
                                 b_sum[l] / np.maximum(b_cnt[l], 1), 0.0)
            new_weights.append([W_avg, b_avg])
        for lyr, wb in zip(dense, new_weights):
            lyr.set_weights(wb)

        for l in range(NUM_HIDDEN_LAYERS):
            touched = luil_cnt[l] > 0
            avg_new = np.where(touched, luil_sum[l] / np.maximum(luil_cnt[l], 1), 0.0)
            uil_global[l] = np.where(touched, avg_new, uil_global[l])

        cov = [float((W_cnt[l] > 0).mean()) for l in range(NUM_HIDDEN_LAYERS + 1)]
        units_cov = [float((b_cnt[l] > 0).mean()) for l in range(NUM_HIDDEN_LAYERS)]
        print(f"  [R{round_id}] cell coverage by layer: "
              f"{['%.2f' % c for c in cov]} | units: "
              f"{['%.2f' % c for c in units_cov]}")

        server_model._halo_layer_ratio = list(np.mean(np.asarray(round_ratios), axis=0))
        return uil_global, cov, units_cov

    server_model = build_server_model_keras(input_dim, n_targets)
    uil_global = init_uil_from_norms(server_model)
    server_model._halo_layer_ratio = [1.0] * NUM_HIDDEN_LAYERS
    coverage_log = []
    for r in range(1, HALO_ROUNDS + 1):
        print(f"\n  {'=' * 12} ROUND HALOFL {r}/{HALO_ROUNDS} {'=' * 12}")
        uil_global, cov, units_cov = halofl_round(server_model, uil_global, r)
        coverage_log.append((r, cov, units_cov))
        print_round_progress("HaloFL", r, HALO_ROUNDS,
                             lambda X: server_model.predict(X, verbose=0), clients)

    ever = [np.zeros(h, dtype=bool) for h in GLOBAL_HIDDEN_SIZES]
    print("\n  Unit coverage by layer (final round):",
          [f"{c:.2f}" for c in coverage_log[-1][2]])

    results_df = evaluate_over_clients(
        lambda X: server_model.predict(X, verbose=0), clients, target_names,
        "HaloFL: final reconstructed global model (mean across clients)",
    )
    save_average_row(results_df, "halofl")
    return results_df

def run_fedgse(data):
    """Run the FedGSE client-selection and partial aggregation pipeline."""
    print("\n" + "#" * 78)
    print("# METHOD 3/4: FedGSE (gradient-based partitioning + partial aggregation)")
    print("#" * 78)

    reseed_random()
    input_dim, n_targets = data["input_dim"], data["n_targets"]
    clients = data["clients"]
    target_names = data["target_names"]
    capacities = FEDGSE_CAPACITIES

    def hidden_width(capacity, full_width):
        return max(1, min(full_width, int(math.ceil(capacity * full_width))))

    def select_neuron_indices_init(global_model, x_client, y_client, capacity):

        dense = get_dense_layers(global_model)
        selected = []
        for l, fw in enumerate(GLOBAL_HIDDEN_SIZES):
            n = hidden_width(capacity, fw)
            norm = np.linalg.norm(dense[l].get_weights()[0], axis=0)
            top = np.argsort(norm)[-n:]
            selected.append(np.sort(top).astype(np.int32))
        return selected

    def _fit_local_generator(x_client, rng):

        x_client = np.asarray(x_client, dtype=np.float64)
        if FEDGSE_SYNTH_GENERATOR == "gmm" and len(x_client) >= 2:
            from sklearn.mixture import GaussianMixture
            n_comp = int(min(FEDGSE_GMM_COMPONENTS, len(x_client)))
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                reg_covar=FEDGSE_SYNTH_COV_REG,
                random_state=int(rng.integers(0, 2 ** 31 - 1)),
            )
            gmm.fit(x_client)

            def sample(n):
                s, _ = gmm.sample(n)
                return s.astype(np.float32)
            return sample

        mean = x_client.mean(axis=0)
        if len(x_client) >= 2:
            cov = np.cov(x_client, rowvar=False)
        else:
            cov = np.eye(x_client.shape[1])
        cov = np.atleast_2d(cov) + FEDGSE_SYNTH_COV_REG * np.eye(x_client.shape[1])

        def sample(n):
            return rng.multivariate_normal(mean, cov, size=n).astype(np.float32)
        return sample

    def build_public_auxiliary():

        rng = np.random.default_rng()
        x_parts, y_parts = [], []

        seed_model = build_server_model_keras(input_dim, n_targets)

        for cid, (cl, capacity) in enumerate(zip(clients, capacities), 1):
            x_client, y_client = cl["X_tr"], cl["y_tr"]

            sampler = _fit_local_generator(x_client, rng)
            n_syn = min(FEDGSE_SYNTH_PER_CLIENT, max(FEDGSE_MIN_SIMILAR, len(x_client)))
            x_syn = sampler(n_syn)

            selected = select_neuron_indices_init(seed_model, x_client, y_client, capacity)
            labeler = build_extracted_submodel(seed_model, selected)
            labeler.fit(x_client, y_client, epochs=FEDGSE_LOCAL_EPOCHS,
                        batch_size=FEDGSE_BATCH_SIZE, verbose=0)
            y_syn = labeler.predict(x_syn, verbose=0).astype(np.float32)

            x_parts.append(x_syn)
            y_parts.append(y_syn)
            print(f"  Client {cid}: generates {n_syn} samples sinteticas "
                  f"({FEDGSE_SYNTH_GENERATOR}) labeled by the local submodel.")

        x_pub = np.concatenate(x_parts, axis=0).astype(np.float32)
        y_pub = np.concatenate(y_parts, axis=0).astype(np.float32)
        print(f"  Synthetic public dataset: {len(x_pub)} samples en total.")
        return x_pub, y_pub

    def select_similar_classification(client_model, x_pub, y_pub):

        pred = np.clip(client_model.predict(x_pub, verbose=0), 1e-8, 1.0)
        err = -np.sum(np.asarray(y_pub) * np.log(pred), axis=1)
        n_keep = min(FEDGSE_MAX_SIMILAR, len(x_pub))
        n_keep = max(min(FEDGSE_MIN_SIMILAR, len(x_pub)), n_keep)
        idx = np.argsort(err)[:n_keep]
        return x_pub[idx], y_pub[idx]

    def activation_gradient_scores(global_model, x_sim, y_sim):
        hidden = get_dense_layers(global_model)[:-1]
        act_model = Model(inputs=global_model.input,
                          outputs=[l.output for l in hidden] + [global_model.output])
        x_t = tf.convert_to_tensor(x_sim, dtype=tf.float32)
        y_t = tf.convert_to_tensor(y_sim, dtype=tf.float32)
        with tf.GradientTape() as tape:
            outs = act_model(x_t, training=False)
            hidden_acts, pred = outs[:-1], outs[-1]

            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_t, pred))
        grads = tape.gradient(loss, hidden_acts)
        return [tf.reduce_mean(tf.abs(g), axis=0).numpy() for g in grads]

    def select_neuron_indices(global_model, x_sim, y_sim, capacity):
        scores = activation_gradient_scores(global_model, x_sim, y_sim)
        selected = []
        for layer_id, (ls, fw) in enumerate(zip(scores, GLOBAL_HIDDEN_SIZES), start=1):
            n = hidden_width(capacity, fw)
            top = np.argpartition(ls, -n)[-n:]
            selected.append(np.sort(top).astype(np.int32))
        return selected

    def build_extracted_submodel(global_model, selected):
        m = Sequential([Input(shape=(input_dim,))])
        for indices in selected:
            m.add(Dense(len(indices), activation="relu"))
        m.add(Dense(n_targets, activation="softmax"))
        m.compile(optimizer=Adam(LEARNING_RATE), loss="categorical_crossentropy")
        g_dense, c_dense = get_dense_layers(global_model), get_dense_layers(m)
        previous = np.arange(input_dim)
        for layer_id, current in enumerate(selected):
            gw, gb = g_dense[layer_id].get_weights()
            c_dense[layer_id].set_weights([gw[np.ix_(previous, current)], gb[current]])
            previous = current
        ow, ob = g_dense[-1].get_weights()
        c_dense[-1].set_weights([ow[previous, :], ob.copy()])
        return m

    def aggregate_partial(global_model, local_models, masks, sample_counts):
        g_dense = get_dense_layers(global_model)
        total = float(sum(sample_counts))
        old = [lyr.get_weights() for lyr in g_dense]
        new = [[w.copy(), b.copy()] for w, b in old]
        for cmodel, selected, ns in zip(local_models, masks, sample_counts):
            coef = ns / total
            l_dense = get_dense_layers(cmodel)
            previous = np.arange(old[0][0].shape[0])
            for layer_id, current in enumerate(selected):
                lw, lb = l_dense[layer_id].get_weights()
                ow, ob = old[layer_id]
                rc = np.ix_(previous, current)
                new[layer_id][0][rc] += coef * (lw - ow[rc])
                new[layer_id][1][current] += coef * (lb - ob[current])
                previous = current
            lw, lb = l_dense[-1].get_weights()
            ow, ob = old[-1]
            ob_block = np.ix_(previous, np.arange(ow.shape[1]))
            new[-1][0][ob_block] += coef * (lw - ow[ob_block])
            new[-1][1] += coef * (lb - ob)
        for lyr, (w, b) in zip(g_dense, new):
            lyr.set_weights([w, b])

    x_pub, y_pub = build_public_auxiliary()
    fedgse_model = build_server_model_keras(input_dim, n_targets)
    prev_client_models = [None] * NUM_CLIENTS

    for round_id in range(1, FEDGSE_ROUNDS + 1):
        print(f"\n  {'=' * 12} ROUND FEDGSE {round_id}/{FEDGSE_ROUNDS} {'=' * 12}")
        local_models, masks, sample_counts = [], [], []
        for cid, (cl, capacity) in enumerate(zip(clients, capacities), 1):
            x_client, y_client = cl["X_tr"], cl["y_tr"]
            if prev_client_models[cid - 1] is None:
                x_sim, y_sim = x_pub, y_pub
            else:
                x_sim, y_sim = select_similar_classification(
                    prev_client_models[cid - 1], x_pub, y_pub)
            selected = select_neuron_indices(fedgse_model, x_sim, y_sim, capacity)
            submodel = build_extracted_submodel(fedgse_model, selected)
            submodel.fit(x_client, y_client, epochs=FEDGSE_LOCAL_EPOCHS,
                         batch_size=FEDGSE_BATCH_SIZE, verbose=0)
            local_models.append(submodel)
            masks.append(selected)
            sample_counts.append(len(x_client))
            print(f"    client {cid} | capacity={capacity:.4f} "
                  f"| units={[len(s) for s in selected]}")

        aggregate_partial(fedgse_model, local_models, masks, sample_counts)
        prev_client_models = local_models
        print_round_progress("FedGSE", round_id, FEDGSE_ROUNDS,
                             lambda X: fedgse_model.predict(X, verbose=0), clients)

    results_df = evaluate_over_clients(
        lambda X: fedgse_model.predict(X, verbose=0), clients, target_names,
        "FedGSE: final reconstructed global model (mean across clients)",
    )
    save_average_row(results_df, "fedgse")
    return results_df

def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    return (z > 0.0).astype(z.dtype)

def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

class MLP:
    """NumPy MLP used by the SubFLOT implementation."""

    def __init__(self, in_dim, hidden, out_dim, seed=None):
        self.in_dim = in_dim
        self.hidden = list(hidden)
        self.out_dim = out_dim
        self.dims = [in_dim] + list(hidden) + [out_dim]

        r = np.random.default_rng(seed)
        self.W, self.b = [], []
        for a, c in zip(self.dims[:-1], self.dims[1:]):
            limit = np.sqrt(6.0 / (a + c))
            self.W.append(r.uniform(-limit, limit, size=(a, c)).astype(np.float64))
            self.b.append(np.zeros(c, dtype=np.float64))
        self.n_layers = len(self.W)
        self._init_adam()

    def _init_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X, cache=False):
        a = X
        zs, as_ = [], [X]
        for l in range(self.n_layers):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = relu(z) if l < self.n_layers - 1 else softmax(z)
            as_.append(a)
        return (a, zs, as_) if cache else a

    def predict(self, X):

        return self.forward(X)

    def train_step(self, Xb, Yb, lr, anchor=None, mu=0.0):
        n = Xb.shape[0]
        out, zs, as_ = self.forward(Xb, cache=True)

        dout = (out - Yb) / n
        gW = [None] * self.n_layers
        gb = [None] * self.n_layers
        delta = dout
        for l in reversed(range(self.n_layers)):
            gW[l] = as_[l].T @ delta
            gb[l] = delta.sum(axis=0)
            if l > 0:
                da = delta @ self.W[l].T
                delta = da * relu_grad(zs[l - 1])
        if anchor is not None and mu > 0.0:
            for l in range(self.n_layers):
                gW[l] = gW[l] + mu * (self.W[l] - anchor.W[l])
                gb[l] = gb[l] + mu * (self.b[l] - anchor.b[l])
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for l in range(self.n_layers):
            self.mW[l] = b1 * self.mW[l] + (1 - b1) * gW[l]
            self.vW[l] = b2 * self.vW[l] + (1 - b2) * (gW[l] ** 2)
            self.W[l] -= lr * (self.mW[l] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vW[l] / (1 - b2 ** self.t)) + eps)
            self.mb[l] = b1 * self.mb[l] + (1 - b1) * gb[l]
            self.vb[l] = b2 * self.vb[l] + (1 - b2) * (gb[l] ** 2)
            self.b[l] -= lr * (self.mb[l] / (1 - b1 ** self.t)) / (
                np.sqrt(self.vb[l] / (1 - b2 ** self.t)) + eps)

    def fit(self, X, Y, epochs, lr, batch=64, anchor=None, mu=0.0, seed=None):

        r = np.random.default_rng(seed)
        N = X.shape[0]
        for _ in range(epochs):
            idx = r.permutation(N)
            for s in range(0, N, batch):
                bi = idx[s:s + batch]
                self.train_step(X[bi], Y[bi], lr, anchor=anchor, mu=mu)

    def copy(self):
        m = MLP(self.in_dim, self.hidden, self.out_dim)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        m._init_adam()
        return m

def _pruning_rate(client_widths):
    per_layer = [1.0 - cw / sw for cw, sw in zip(client_widths, GLOBAL_HIDDEN_SIZES)]
    return per_layer, float(np.mean(per_layer))

def _emd(mu, nu, M):

    try:
        import ot as pot
        return pot.emd(np.ascontiguousarray(mu), np.ascontiguousarray(nu),
                       np.ascontiguousarray(M))
    except ImportError:
        pass
    n, m = M.shape
    A_eq, b_eq = [], []
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1.0
        A_eq.append(row)
        b_eq.append(mu[i])
    for j in range(m):
        col = np.zeros(n * m)
        col[j::m] = 1.0
        A_eq.append(col)
        b_eq.append(nu[j])
    res = linprog(M.ravel(), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"EMD LP fallo: {res.message}")
    return res.x.reshape(n, m)

def _sinkhorn(mu, nu, M, reg=None, n_iter=1000, tol=1e-9):
    reg = SUBFLOT_SINKHORN_REG if reg is None else reg
    K = np.exp(-M / reg) + 1e-300
    u, v = np.ones_like(mu), np.ones_like(nu)
    for _ in range(n_iter):
        u_prev = u
        u = mu / (K @ v + 1e-300)
        v = nu / (K.T @ u + 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]

def _solve_ot(mu, nu, M):
    return _emd(mu, nu, M) if SUBFLOT_OT_SOLVER == "emd" else _sinkhorn(mu, nu, M)

def _normalize_otmap_sum(T):

    col = T.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    return T / col

def _normed_vecs(V, eps=1e-9):

    return V / (np.linalg.norm(V, axis=-1, keepdims=True) + eps)

def _cost_matrix(A, B):

    A, B = _normed_vecs(A), _normed_vecs(B)
    a2 = (A ** 2).sum(axis=1, keepdims=True)
    b2 = (B ** 2).sum(axis=1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)
    return d2 / (d2.sum() + 1e-12)

def _ot_align_mlp(src_model, tgt_model, mix_ratio=SUBFLOT_MIX_RATIO):
    """Align source neurons to the target architecture using optimal transport."""

    aligned = tgt_model.copy()
    t_in = None
    n = src_model.n_layers
    for l in range(n):
        src_W, src_b = src_model.W[l], src_model.b[l]
        src_in_aligned = src_W if t_in is None else t_in.T @ src_W

        if l == n - 1:

            if not SUBFLOT_SKIP_HEAD:
                aligned.W[l] = (mix_ratio * src_in_aligned
                                + (1 - mix_ratio) * tgt_model.W[l]).copy()
                aligned.b[l] = (mix_ratio * src_b
                                + (1 - mix_ratio) * tgt_model.b[l]).copy()
            break

        src_neurons, tgt_neurons = src_in_aligned.T, tgt_model.W[l].T
        mu = np.ones(src_neurons.shape[0]) / src_neurons.shape[0]
        nu = np.ones(tgt_neurons.shape[0]) / tgt_neurons.shape[0]
        T = _normalize_otmap_sum(
            _solve_ot(mu, nu, _cost_matrix(src_neurons, tgt_neurons)))

        W_aligned = src_in_aligned @ T
        b_aligned = T.T @ src_b
        aligned.W[l] = (mix_ratio * W_aligned
                        + (1 - mix_ratio) * tgt_model.W[l]).copy()
        aligned.b[l] = (mix_ratio * b_aligned
                        + (1 - mix_ratio) * tgt_model.b[l]).copy()
        t_in = T
    aligned._init_adam()
    return aligned

def _prune_to_widths(global_model, widths, seed=None):

    kept = [np.arange(global_model.in_dim)]
    for l, w in enumerate(widths):
        kept.append(np.arange(w))
    kept.append(np.arange(global_model.out_dim))
    sub = MLP(global_model.in_dim, widths, global_model.out_dim, seed=seed)
    for l in range(global_model.n_layers):
        sub.W[l] = global_model.W[l][np.ix_(kept[l], kept[l + 1])].copy()
        sub.b[l] = global_model.b[l][kept[l + 1]].copy()
    sub._init_adam()
    return sub

class SubFLOTClient:
    def __init__(self, cid, Xc, Yc, widths):
        self.cid = cid
        self.Xc = Xc
        self.Yc = Yc
        self.widths = widths
        self.train_samples = Xc.shape[0]
        _, self.rho = _pruning_rate(widths)
        self.model = None
        self.anchor = None

    def set_model(self, submodel):
        self.model = submodel.copy()

    def update_model(self, anchor_model):
        self.anchor = anchor_model.copy()
        self.model = anchor_model.copy()

    def train(self):
        mu = SUBFLOT_SAR_LAMBDA * (self.rho if SUBFLOT_DYNAMIC_MU else 1.0)
        self.model.fit(self.Xc, self.Yc, epochs=SUBFLOT_LOCAL_EPOCHS, lr=LEARNING_RATE,
                       batch=SUBFLOT_BATCH_SIZE, anchor=self.anchor, mu=mu,
                       seed=rand_seed())

class SubFLOTServer:
    def __init__(self, client_data_widths, in_dim, out_dim, X_te_v, y_te_v):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.global_model = MLP(in_dim, GLOBAL_HIDDEN_SIZES, out_dim, seed=rand_seed())
        self.X_te_v, self.y_te_v = X_te_v, y_te_v
        self.clients = []
        for cid, (Xc, Yc, w) in enumerate(client_data_widths):
            c = SubFLOTClient(cid, Xc, Yc, w)
            c.set_model(_prune_to_widths(self.global_model, w, seed=rand_seed()))
            self.clients.append(c)

    def otp_generate_submodel(self, cid):

        return _ot_align_mlp(self.global_model, self.clients[cid].model,
                             mix_ratio=SUBFLOT_MIX_RATIO)

    def send_models(self):
        for c in self.clients:
            c.update_model(self.otp_generate_submodel(c.cid))

    def send_models_warmup(self):
        for c in self.clients:
            c.update_model(_prune_to_widths(self.global_model, c.widths,
                                            seed=rand_seed()))

    def ota_aggregate(self):

        weights = np.array([c.train_samples for c in self.clients], dtype=np.float64)
        weights /= weights.sum()
        acc_W = [np.zeros_like(w) for w in self.global_model.W]
        acc_b = [np.zeros_like(b) for b in self.global_model.b]
        for c, p in zip(self.clients, weights):
            mapped = _ot_align_mlp(c.model, self.global_model,
                                   mix_ratio=SUBFLOT_MIX_RATIO)
            for l in range(self.global_model.n_layers):
                acc_W[l] += p * mapped.W[l]
                acc_b[l] += p * mapped.b[l]
        self.global_model.W = acc_W
        self.global_model.b = acc_b
        self.global_model._init_adam()

    def train(self, monitor_clients=None):
        for rnd in range(SUBFLOT_GLOBAL_ROUNDS):
            print(f"\n  {'=' * 12} ROUND SUBFLOT "
                  f"{rnd + 1}/{SUBFLOT_GLOBAL_ROUNDS} {'=' * 12}")
            if rnd < SUBFLOT_WARMUP_ROUNDS:
                self.send_models_warmup()
            else:
                self.send_models()
            for c in self.clients:
                c.train()
            self.ota_aggregate()
            if monitor_clients is not None:
                print_round_progress(
                    "SubFLOT", rnd + 1, SUBFLOT_GLOBAL_ROUNDS,
                    lambda X: self.global_model.predict(np.asarray(X, dtype=np.float64)),
                    monitor_clients)

    def predict_local(self, cid, X):

        return self.clients[cid].model.predict(np.asarray(X, dtype=np.float64))

def run_subflot(data):
    """Run the SubFLOT OTP/SAR/OTA federated pipeline."""
    print("\n" + "#" * 78)
    print("# METHOD 4/4: SubFLOT (L1 pruning + OTP + SAR + OTA, NumPy backend)")
    print("#" * 78)
    reseed_random()
    input_dim, n_targets = data["input_dim"], data["n_targets"]
    clients = data["clients"]
    target_names = data["target_names"]

    client_data_widths = []
    for cid, (cl, widths) in enumerate(zip(clients, CLIENT_WIDTHS)):
        Xc = cl["X_tr"].astype(np.float64)
        Yc = cl["y_tr"].astype(np.float64)
        client_data_widths.append((Xc, Yc, widths))

    X_te_union = np.concatenate([cl["X_te"] for cl in clients], axis=0).astype(np.float64)
    y_te_union = np.concatenate([cl["y_te"] for cl in clients], axis=0).astype(np.float64)

    server = SubFLOTServer(client_data_widths, input_dim, n_targets, X_te_union, y_te_union)
    server.train(monitor_clients=clients)

    local_accs = [
        accuracy_score(_to_class_indices(cl["y_te"]),
                       _to_class_indices(server.predict_local(cid, cl["X_te"])))
        for cid, cl in enumerate(clients)
    ]
    print(f"\n  [SubFLOT] mean LOCAL-model accuracy = "
          f"{np.mean(local_accs):.4f} {[round(a, 3) for a in local_accs]}")

    results_df = evaluate_over_clients(
        lambda X: server.global_model.predict(np.asarray(X, dtype=np.float64)),
        clients, target_names,
        "SubFLOT: final fused global model (mean across clients)",
    )
    save_average_row(results_df, "subflot")
    return results_df

def run_fedavg(data):
    """Run full-model FedAvg under the same federated training budget."""
    print("\n" + "#" * 78)
    print("# BASELINE: FedAvg (FULL model on every client)")
    print("#" * 78)
    reseed_random()
    input_dim, n_targets = data["input_dim"], data["n_targets"]
    clients = data["clients"]
    target_names = data["target_names"]

    global_model = build_server_model_keras(input_dim, n_targets)

    sample_counts = [len(cl["X_tr"]) for cl in clients]
    total_samples = float(sum(sample_counts))
    coefs = [n / total_samples for n in sample_counts]

    local_model = build_server_model_keras(input_dim, n_targets)

    for round_id in range(1, FEDAVG_ROUNDS + 1):
        print(f"\n  {'=' * 12} ROUND FEDAVG {round_id}/{FEDAVG_ROUNDS} {'=' * 12}")
        global_weights = global_model.get_weights()
        acc = [np.zeros_like(w) for w in global_weights]

        for cid, (cl, coef) in enumerate(zip(clients, coefs), 1):

            local_model.set_weights(global_weights)

            local_model.compile(optimizer=Adam(LEARNING_RATE),
                                loss="categorical_crossentropy")

            local_model.fit(cl["X_tr"], cl["y_tr"], epochs=FEDAVG_LOCAL_EPOCHS,
                            batch_size=FEDAVG_BATCH_SIZE, verbose=0)

            for k, w in enumerate(local_model.get_weights()):
                acc[k] += coef * w
            print(f"    client {cid} | full model {GLOBAL_HIDDEN_SIZES} "
                  f"| n={len(cl['X_tr'])} | weight={coef:.4f}")

        global_model.set_weights(acc)
        print_round_progress("FedAvg", round_id, FEDAVG_ROUNDS,
                             lambda X: global_model.predict(X, verbose=0), clients)

    results_df = evaluate_over_clients(
        lambda X: global_model.predict(X, verbose=0), clients, target_names,
        "FedAvg: final global model (mean across clients)",
    )
    save_average_row(results_df, "fedavg")
    return results_df

EXTRA_RESULTS = {}

METHODS = {"stitchfl": run_stitchfl, "halofl": run_halofl, "fedgse": run_fedgse,
           "subflot": run_subflot, "fedavg": run_fedavg}

def parse_args():
    p = argparse.ArgumentParser(
        description="Federated pipeline for UCI Letter Recognition. "
                    "Runs StitchFL only by default.")
    p.add_argument("--methods", "--metodos", dest="methods", nargs="+", default=["stitchfl"],
                   choices=list(METHODS) + ["all", "todos"])
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--stitch-mode", "--cosido", dest="stitch_mode", default=STITCH_MODE, choices=["factor", "pinv", "off"],
                   help="skip-edge stitching strategy (default: factor)")
    p.add_argument("--distill-scope", "--alcance", dest="distill_scope", default=STITCH_DISTILL_SCOPE, choices=["all", "head"],
                   help="distill the full model or only the output head")
    p.add_argument("--anchors", "--anclas", dest="anchors", type=int, default=STITCH_N_ANCHOR)
    p.add_argument("--no-recalibration", "--sin-recal", dest="no_recalibration", action="store_true")
    p.add_argument("--no-ensemble", "--sin-ensemble", dest="no_ensemble", action="store_true")
    return p.parse_args()

def validate_stitching_layout():
    """Report whether the configured ranges can support analytical stitching."""
    skipping_clients = [i for i, r in enumerate(CLIENT_RANGES)
              if any(e <= s for (s, e) in r)]
    if not skipping_clients:
        print("WARNING: no client skips a layer in CLIENT_RANGES, so there are no "
              "skip edges and the stitching mode has no effect; aggregation reduces "
              "to cell-wise averaging.")
        return
    for k in range(1, NUM_HIDDEN_LAYERS + 1):
        used_neurons = np.zeros(GLOBAL_HIDDEN_SIZES[k - 1], dtype=bool)
        for r in CLIENT_RANGES:
            used_neurons[r[k - 1][0]:r[k - 1][1]] = True
        free_neurons = int((~used_neurons).sum())
        required_neurons = sum(r[k - 2][1] - r[k - 2][0] for i, r in enumerate(CLIENT_RANGES)
                         if i in skipping_clients and k >= 2 and r[k - 1][1] <= r[k - 1][0])
        if required_neurons:
            status = "SUFFICIENT" if free_neurons >= required_neurons else "INSUFFICIENT"
            print(f"WARNING: layer {k} | free neurons {free_neurons} | required for routing "
                  f"{required_neurons} -> {status}. With 'factor', exact stitching "
                  f"requires sufficient capacity; increase GLOBAL_HIDDEN_SIZES[{k - 1}].")

def main():
    args = parse_args()
    selected_methods = list(METHODS) if any(x in args.methods for x in ("all", "todos")) else args.methods
    globals()["STITCH_MODE"] = args.stitch_mode
    globals()["STITCH_DISTILL_SCOPE"] = args.distill_scope
    globals()["STITCH_N_ANCHOR"] = args.anchors
    globals()["STITCH_RECALIBRATE"] = not args.no_recalibration
    globals()["STITCH_EVAL_ENSEMBLE"] = not args.no_ensemble

    all_results = []
    for rep in range(args.reps):
        if args.reps > 1:
            print("\n" + "#" * 78 + f"\n# RUN {rep + 1}/{args.reps}\n" + "#" * 78)
        reseed_random()
        print("Loading Letter Recognition and partitioning into "
              f"{NUM_CLIENTS} clients (80/20 split per client)...")
        data = load_data_and_clients()
        print(f"\ninput_dim={data['input_dim']} | n_classes={data['n_targets']} "
              f"| clients={NUM_CLIENTS} | architecture={GLOBAL_HIDDEN_SIZES}")
        print(f"Metrics: {', '.join(METRIC_COLUMNS)} "
              f"(Precision/Recall/F1 use '{METRIC_AVERAGE}' averaging)")
        print(f"Results directory: {RESULTS_DIR}")
        print(f"Federated setup: {FL_ROUNDS} rounds x {FL_LOCAL_EPOCHS} "
              f"local epochs | final evaluation only")
        print(f"Methods: {', '.join(selected_methods)}")
        if "stitchfl" in selected_methods:
            print(f"StitchFL | stitching '{STITCH_MODE}' | distillation: "
                  + (f"Gaussian anchors x{STITCH_N_ANCHOR}, scope "
                     f"'{STITCH_DISTILL_SCOPE}'" if STITCH_RECALIBRATE
                     else "DISABLED"))
            validate_stitching_layout()

        results = {}
        EXTRA_RESULTS.clear()
        for method_name in selected_methods:
            results[method_name] = METHODS[method_name](data)
        results.update(EXTRA_RESULTS)

        for method_name, df in results.items():
            row = df.loc[df["Output"] == "Average"].iloc[0].to_dict()
            row.update({"Method": method_name, "Rep": rep + 1})
            all_results.append(row)

    print("\n" + "=" * 78)
    print(f"FINAL SUMMARY ({FL_ROUNDS} rounds x {FL_LOCAL_EPOCHS} local epochs)")
    print("   ens_logit_clientes is a reference ensemble, not a federated method; "
          "it requires all five submodels at inference.")
    print("=" * 78)
    df_res = pd.DataFrame(all_results)[["Rep", "Method"] + METRIC_COLUMNS]
    print(tabulate(df_res, headers="keys", tablefmt="github",
                   floatfmt=".4f", showindex=False))
    if args.reps > 1:
        print("\nMean and standard deviation across runs:")
        agg = df_res.groupby("Method")[METRIC_COLUMNS].agg(["mean", "std"]).round(4)
        agg.columns = [f"{m}_{st}" for m, st in agg.columns]
        print(tabulate(agg.reset_index(), headers="keys", tablefmt="github",
                       floatfmt=".4f", showindex=False))

if __name__ == "__main__":
    main()
