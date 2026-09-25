"""Federated CNN benchmark for heterogeneous submodel training on CIFAR-10.

Fusion methods and FedAvg use the same federated training budget and are
evaluated on client-local test splits only after the final communication round.
"""

import argparse
import gc
import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

try:
    from fusions_fl import FUSIONS
    _FUSIONS_MODULE = "fusions_fl"
except ImportError:
    from fusions import FUSIONS
    _FUSIONS_MODULE = "fusions"

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(df, headers="keys", tablefmt=None, floatfmt=".4f", showindex=True):
        return df.to_string(index=showindex)

# CIFAR-10 normalization.
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# Experiment configuration.
BASE_DIR = pathlib.Path(os.getcwd())
DATA_DIR = BASE_DIR / "Datasets" / "cifar"
DATA_PATTERN = str(DATA_DIR / "cifar10_part_{}.csv")
TEST_FRACTION = 0.1
RESULTS_DIR = BASE_DIR / "results" / "CNN"

GLOBAL_CONV_CHANNELS = [64, 128, 256]
GLOBAL_DENSE_UNITS = [256]
NUM_CLASSES = 10

FL_ROUNDS = 30
FL_LOCAL_EPOCHS = 5

MONITOR_TRAIN_SAMPLES = 2000

CLIENT_CONV_RANGES = [
    [(0, 24),   (0, 48),    (0, 96)],
    [(12, 36),  (24, 72),   (48, 144)],
    [(24, 48),  (48, 96),   (96, 192)],
    [(36, 56),  (72, 112),  (144, 224)],
    [(44, 64),  (88, 128),  (176, 256)],
]
CLIENT_DENSE_RANGES = [
    [(0, 96)],
    [(48, 160)],
    [(96, 192)],
    [(144, 224)],
    [(160, 256)],
]

NUM_CLIENTS = len(CLIENT_CONV_RANGES)
RANDOM_SEED = 42

# Data loading.
def load_csv(path, channels_first=False):
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    label_col = None
    for cand in ("label", "labels", "class", "target", "y"):
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break
    if label_col is not None:
        y = df[label_col].to_numpy().astype(np.int64)
        pixels = df.drop(columns=[label_col]).to_numpy().astype(np.float32)
    else:
        values = df.to_numpy()
        y = values[:, 0].astype(np.int64)
        pixels = values[:, 1:].astype(np.float32)
    if pixels.shape[1] != 3072:
        raise ValueError(f"{path}: expected 3072 pixels per row, found {pixels.shape[1]}")
    if pixels.max() > 1.5:
        pixels /= 255.0
    if channels_first:
        pixels = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    else:
        pixels = pixels.reshape(-1, 32, 32, 3)
    pixels = (pixels - CIFAR10_MEAN) / CIFAR10_STD
    return pixels.astype(np.float32), y

def build_client_paths(pattern, clients):
    paths = [pattern.format(i) for i in range(1, clients + 1)]
    for client_id, path in enumerate(paths, 1):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} (expected dataset for client {client_id})")
    return paths

def load_client(path, channels_first):
    return load_csv(path, channels_first)

def split_train_test(x, y, frac, rng):
    num_test_samples = max(1, int(frac * len(x)))
    permutation = rng.permutation(len(x))
    test_indices, train_indices = permutation[:num_test_samples], permutation[num_test_samples:]
    return (x[train_indices], y[train_indices]), (x[test_indices], y[test_indices])

# Model construction.
def build_cnn(conv_channels, dense_hidden, n_classes=NUM_CLASSES, lr=1e-3, name="cnn"):
    m = Sequential(name=name)
    m.add(layers.Input(shape=(32, 32, 3)))
    for i, channels in enumerate(conv_channels):
        m.add(layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"conv{i}"))
        m.add(layers.BatchNormalization(name=f"bn{i}"))
        m.add(layers.Activation("relu"))
        m.add(layers.MaxPooling2D())
        m.add(layers.Dropout(0.25 + 0.1 * i))
    m.add(layers.GlobalAveragePooling2D())
    for j, h in enumerate(dense_hidden):
        m.add(layers.Dense(h, activation="relu", name=f"dense{j}"))
        m.add(layers.Dropout(0.5))
    m.add(layers.Dense(n_classes, name="out"))
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m

def build_server_model(lr=1e-3, name="server_cnn"):
    return build_cnn(GLOBAL_CONV_CHANNELS, GLOBAL_DENSE_UNITS, lr=lr, name=name)

def build_client_submodel(conv_ranges, dense_ranges, lr=1e-3, name="client_cnn"):
    channels = [e - s for (s, e) in conv_ranges]
    hidden_units = [e - s for (s, e) in dense_ranges if e > s]
    return build_cnn(channels, hidden_units, lr=lr, name=name)

def compute_accuracy(model, x, y):
    return float((model.predict(x, verbose=0).argmax(1) == y).mean())

# Evaluation.
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "Specificity",
                  "Balanced_Acc", "MCC", "AUC_ovr", "LogLoss", "Support"]

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def evaluate_metrics(model, x, y, label=""):

    logits = model.predict(x, verbose=0)
    proba = softmax(logits.astype(np.float64))
    y_pred = logits.argmax(1)
    acc_global = float((y_pred == y).mean())

    records = []
    for c in range(NUM_CLASSES):
        pos = (y == c)
        pred_pos = (y_pred == c)
        tp = int(np.sum(pred_pos & pos))
        fp = int(np.sum(pred_pos & ~pos))
        fn = int(np.sum(~pred_pos & pos))
        tn = int(np.sum(~pred_pos & ~pos))

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bacc = 0.5 * (rec + spec)

        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / den) if den > 0 else 0.0

        s = proba[:, c]
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos and n_neg:
            order = np.argsort(s)
            ranks = np.empty(len(s), dtype=np.float64)
            ranks[order] = np.arange(1, len(s) + 1)
            auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        else:
            auc = np.nan

        eps = 1e-12
        ll_c = -np.mean(np.log(np.clip(s[pos], eps, 1.0))) if n_pos else np.nan

        records.append({
            "Clase": c, "Accuracy": rec, "Precision": prec, "Recall": rec,
            "F1": f1, "Specificity": spec, "Balanced_Acc": bacc, "MCC": mcc,
            "AUC_ovr": auc, "LogLoss": ll_c, "Support": int(pos.sum()),
        })

    df = pd.DataFrame(records)
    avg = df[METRIC_COLUMNS].mean(axis=0, skipna=True).to_dict()
    avg["Support"] = int(len(y))
    avg["Accuracy"] = acc_global
    df = pd.concat([df, pd.DataFrame([{"Clase": "Average", **avg}])],
                   ignore_index=True)
    if label:
        print(f"  [{label}] accuracy global = {acc_global*100:.2f}%")
    return df, acc_global

def evaluate_over_clients(model, test_sets, method, label):

    frames, accs = [], []
    for i, (xte, yte) in enumerate(test_sets, 1):
        df_i, acc = evaluate_metrics(model, xte, yte, f"{label} test C{i}")
        df_i.insert(0, "Test", f"Client {i}")
        df_i.insert(0, "Metodo", method)
        frames.append(df_i)
        accs.append(acc)
    return frames, accs

def append_summary_row(method_name, per_client_frames, extra=None):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{method_name}.csv"

    avgs = [df[df["Clase"] == "Average"].iloc[0] for df in per_client_frames]
    row = {"Output": "Average"}
    for col in METRIC_COLUMNS:
        vals = [float(a.get(col)) for a in avgs if a.get(col) is not None]
        row[col] = float(np.mean(vals)) if vals else None
    if extra:
        row.update(extra)

    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)
    print(f"    appended row to: {path}")

# Submodel extraction and coverage.
def get_dense_layers(m):
    return [l for l in m.layers if isinstance(l, layers.Dense)]

def get_conv_layers(m):
    return [l for l in m.layers if isinstance(l, layers.Conv2D)]

def get_batchnorm_layers(m):
    return [l for l in m.layers if isinstance(l, layers.BatchNormalization)]

def check_no_layer_skips():

    for i, dense_ranges in enumerate(CLIENT_DENSE_RANGES, 1):
        if any(e <= s for (s, e) in dense_ranges):
            raise ValueError(
                f"Client {i} skips a dense layer ({dense_ranges}). Round-based federated "
                f"training requires all ranges to satisfy end > start."
            )

def extract_client_submodel(server, conv_ranges, dense_ranges, lr, name):

    sub = build_client_submodel(conv_ranges, dense_ranges, lr, name)
    s_conv, s_bn, s_dense = get_conv_layers(server), get_batchnorm_layers(server), get_dense_layers(server)
    c_conv, c_bn, c_dense = get_conv_layers(sub), get_batchnorm_layers(sub), get_dense_layers(sub)

    for layer_idx, (channel_start, channel_end) in enumerate(conv_ranges):
        W = s_conv[layer_idx].get_weights()[0]
        if layer_idx == 0:
            k = W[:, :, :, channel_start:channel_end]
        else:
            prev_start, prev_end = conv_ranges[layer_idx - 1]
            k = W[:, :, prev_start:prev_end, channel_start:channel_end]
        c_conv[layer_idx].set_weights([k.copy()])
        c_bn[layer_idx].set_weights([w[channel_start:channel_end].copy() for w in s_bn[layer_idx].get_weights()])

    prev_s, prev_e = conv_ranges[-1]
    for dense_idx, (dense_start, dense_end) in enumerate(dense_ranges):
        Wd, bd = s_dense[dense_idx].get_weights()
        c_dense[dense_idx].set_weights([Wd[prev_s:prev_e, dense_start:dense_end].copy(), bd[dense_start:dense_end].copy()])
        prev_s, prev_e = dense_start, dense_end

    Wo, bo = s_dense[-1].get_weights()
    c_dense[-1].set_weights([Wo[prev_s:prev_e, :].copy(), bo.copy()])
    return sub

def coverage_masks(server):

    s_conv, s_bn, s_dense = get_conv_layers(server), get_batchnorm_layers(server), get_dense_layers(server)
    masks = {"conv": [], "bn": [], "dense": [], "out": None}

    for layer_idx, conv in enumerate(s_conv):
        W = conv.get_weights()[0]
        m = np.zeros(W.shape[2:4], dtype=bool)
        for conv_ranges in CLIENT_CONV_RANGES:
            channel_start, channel_end = conv_ranges[layer_idx]
            if layer_idx == 0:
                m[:, channel_start:channel_end] = True
            else:
                prev_start, prev_end = conv_ranges[layer_idx - 1]
                m[prev_start:prev_end, channel_start:channel_end] = True
        masks["conv"].append(m)

    for layer_idx, bn in enumerate(s_bn):
        m = np.zeros(bn.get_weights()[0].shape, dtype=bool)
        for conv_ranges in CLIENT_CONV_RANGES:
            channel_start, channel_end = conv_ranges[layer_idx]
            m[channel_start:channel_end] = True
        masks["bn"].append(m)

    n_hidden = len(s_dense) - 1
    for dense_idx in range(n_hidden):
        Wd = s_dense[dense_idx].get_weights()[0]
        m = np.zeros(Wd.shape, dtype=bool)
        for conv_ranges, dense_ranges in zip(CLIENT_CONV_RANGES, CLIENT_DENSE_RANGES):
            dense_start, dense_end = dense_ranges[dense_idx]
            prev_start, prev_end = conv_ranges[-1] if dense_idx == 0 else dense_ranges[dense_idx - 1]
            m[prev_start:prev_end, dense_start:dense_end] = True
        masks["dense"].append(m)

    Wo = s_dense[-1].get_weights()[0]
    mo = np.zeros(Wo.shape, dtype=bool)
    for dense_ranges in CLIENT_DENSE_RANGES:
        prev_start, prev_end = dense_ranges[-1]
        mo[prev_start:prev_end, :] = True
    masks["out"] = mo
    return masks

def snapshot_weights(model):
    return [[w.copy() for w in l.get_weights()] for l in model.layers]

def restore_uncovered(server, prev_snapshot, masks):

    prev_by_layer = {id(l): w for l, w in zip(server.layers, prev_snapshot)}
    s_conv, s_bn, s_dense = get_conv_layers(server), get_batchnorm_layers(server), get_dense_layers(server)

    for layer_idx, conv in enumerate(s_conv):
        new = conv.get_weights()[0]
        old = prev_by_layer[id(conv)][0]
        m = masks["conv"][layer_idx][None, None, :, :]
        conv.set_weights([np.where(m, new, old).astype(new.dtype)])

    for layer_idx, bn in enumerate(s_bn):
        new_all = bn.get_weights()
        old_all = prev_by_layer[id(bn)]
        m = masks["bn"][layer_idx]
        bn.set_weights([np.where(m, n, o).astype(n.dtype)
                        for n, o in zip(new_all, old_all)])

    for dense_idx in range(len(s_dense) - 1):
        Wn, bn_ = s_dense[dense_idx].get_weights()
        Wo, bo = prev_by_layer[id(s_dense[dense_idx])]
        m = masks["dense"][dense_idx]
        mb = m.any(axis=0)
        s_dense[dense_idx].set_weights([np.where(m, Wn, Wo).astype(Wn.dtype),
                                 np.where(mb, bn_, bo).astype(bn_.dtype)])

    Wn, bn_ = s_dense[-1].get_weights()
    Wo, bo = prev_by_layer[id(s_dense[-1])]
    m = masks["out"]
    s_dense[-1].set_weights([np.where(m, Wn, Wo).astype(Wn.dtype), bn_])

def recalibrate_output_layer(server_model, clients, x_anchor_source, n_anchor=5000):
    rs = np.random.RandomState(12345)
    idx = rs.choice(len(x_anchor_source), size=min(n_anchor, len(x_anchor_source)),
                    replace=False)
    X_anchor = x_anchor_source[idx]
    Y_teacher = np.zeros((len(X_anchor), NUM_CLASSES), dtype=np.float32)
    for c in clients:
        Y_teacher += c.predict(X_anchor, verbose=0).astype(np.float32)
    Y_teacher /= float(len(clients))

    hidden_model = Model(server_model.inputs, server_model.layers[-2].output)
    H = hidden_model.predict(X_anchor, verbose=0).astype(np.float32)
    H_aug = np.hstack([H, np.ones((H.shape[0], 1), dtype=np.float32)])
    A = H_aug.T @ H_aug + 1e-2 * np.eye(H_aug.shape[1], dtype=np.float32)
    Wb = np.linalg.solve(A, H_aug.T @ Y_teacher)
    server_model.layers[-1].set_weights([Wb[:-1, :], Wb[-1, :]])
    print(f"  recalibrated with {len(X_anchor)} anchors")

def monitor_train_accuracy(model, train_sets, n=MONITOR_TRAIN_SAMPLES):

    accs = []
    for xtr, ytr in train_sets:
        k = min(n, len(ytr))
        accs.append(compute_accuracy(model, xtr[:k], ytr[:k]))
    return float(np.mean(accs))

# Federated training loops.
def run_federated_fusion(fusion_name, fusion_fn, train_sets, test_sets, args):

    print(f"\n{'='*70}\nHETERO-FL | fusion {fusion_name} | "
          f"{args.rounds} rounds x {args.local_epochs} epochs\n{'='*70}")
    sample_counts = [len(y) for _, y in train_sets]
    client_x = [x for x, _ in train_sets]

    server = build_server_model(args.lr, name=f"fed_{fusion_name}")
    masks = coverage_masks(server)
    clients = []

    for rnd in range(1, args.rounds + 1):
        print(f"\n  {'='*10} [{fusion_name}] ROUND {rnd}/{args.rounds} {'='*10}")
        for c in clients:
            del c
        clients = []
        gc.collect()

        prev = snapshot_weights(server)
        for i, ((xtr, ytr), conv_ranges, dense_ranges) in enumerate(
                zip(train_sets, CLIENT_CONV_RANGES, CLIENT_DENSE_RANGES), 1):
            channels = [e - s for (s, e) in conv_ranges]
            hidden_units = [e - s for (s, e) in dense_ranges]

            sub = extract_client_submodel(server, conv_ranges, dense_ranges, args.lr,
                                          name=f"{fusion_name}_r{rnd}_c{i}")

            h = sub.fit(xtr, ytr, epochs=args.local_epochs,
                        batch_size=args.batch_size, verbose=args.verbose)
            print(f"    C{i} | channels={channels} dense={hidden_units} | n={len(ytr)} "
                  f"| loss={h.history['loss'][-1]:.4f} "
                  f"acc={h.history['accuracy'][-1]*100:.2f}%")
            clients.append(sub)

        fusion_fn(clients, server, CLIENT_CONV_RANGES, CLIENT_DENSE_RANGES,
              n_classes=NUM_CLASSES, sample_counts=sample_counts,
              client_x=client_x, verbose=(rnd == 1))

        restore_uncovered(server, prev, masks)

        train_accuracy = monitor_train_accuracy(server, train_sets)
        print(f"    [{fusion_name}] Round {rnd}/{args.rounds} | fusion applied "
              f"| mean accuracy (train sample) = {train_accuracy*100:.2f}%")

    submodel_frames, submodel_accuracies_list = [], []
    for i, (c, (xte, yte)) in enumerate(zip(clients, test_sets), 1):
        df_i, acc = evaluate_metrics(c, xte, yte, f"SUB[{fusion_name}] C{i}")
        df_i.insert(0, "Test", f"Client {i}")
        df_i.insert(0, "Metodo", f"SUBMODELOS_{fusion_name}")
        submodel_frames.append(df_i)
        submodel_accuracies_list.append(acc)

    if args.recalibrate and fusion_name == "StitchFL":
        x_all_train = np.concatenate(client_x)
        recalibrate_output_layer(server, clients, x_all_train)

    fusion_frames, fusion_accuracies_list = evaluate_over_clients(server, test_sets, fusion_name, f"FUSION {fusion_name}")
    print(f"  >>> {fusion_name}: mean across clients = {np.mean(fusion_accuracies_list)*100:.2f}%")

    for c in clients:
        del c
    del server
    keras.backend.clear_session()
    gc.collect()
    return fusion_frames, fusion_accuracies_list, submodel_frames, submodel_accuracies_list

def run_fedavg(train_sets, test_sets, args):

    print(f"\n{'='*70}\nFedAvg (full model on all clients) | "
          f"{args.rounds} rounds x {args.local_epochs} epochs\n{'='*70}")
    sample_counts = [len(y) for _, y in train_sets]
    total = float(sum(sample_counts))
    aggregation_weights = [n / total for n in sample_counts]

    global_model = build_server_model(args.lr, name="fedavg_server")
    local_model = build_server_model(args.lr, name="fedavg_local")

    for rnd in range(1, args.rounds + 1):
        print(f"\n  {'='*10} [FedAvg] ROUND {rnd}/{args.rounds} {'='*10}")
        global_weights = global_model.get_weights()
        aggregated_weights = [np.zeros_like(w) for w in global_weights]

        for i, ((xtr, ytr), aggregation_weight) in enumerate(zip(train_sets, aggregation_weights), 1):

            local_model.set_weights(global_weights)

            local_model.compile(
                optimizer=keras.optimizers.Adam(args.lr),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"])

            h = local_model.fit(xtr, ytr, epochs=args.local_epochs,
                                batch_size=args.batch_size, verbose=args.verbose)

            for k, w in enumerate(local_model.get_weights()):
                aggregated_weights[k] += aggregation_weight * w
            print(f"    C{i} | full model {GLOBAL_CONV_CHANNELS}+{GLOBAL_DENSE_UNITS} "
                  f"| n={len(ytr)} weight={aggregation_weight:.4f} "
                  f"| loss={h.history['loss'][-1]:.4f} "
                  f"acc={h.history['accuracy'][-1]*100:.2f}%")

        global_model.set_weights(aggregated_weights)
        train_accuracy = monitor_train_accuracy(global_model, train_sets)
        print(f"    [FedAvg] Round {rnd}/{args.rounds} | fusion applied "
              f"| mean accuracy (train sample) = {train_accuracy*100:.2f}%")

    frames, accs = evaluate_over_clients(global_model, test_sets, "FedAvg", "FedAvg")
    print(f"  >>> FedAvg: mean across clients = {np.mean(accs)*100:.2f}%")

    del local_model, global_model
    keras.backend.clear_session()
    gc.collect()
    return frames, accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=DATA_PATTERN)
    parser.add_argument("--channels-first", action="store_true")
    parser.add_argument("--rounds", type=int, default=FL_ROUNDS,
                    help="federated rounds (server fusion after each round)")
    parser.add_argument("--local-epochs", type=int, default=FL_LOCAL_EPOCHS,
                    help="local training epochs per client and round")
    parser.add_argument("--ref-epochs", type=int, default=None,
                    help="reference-phase epochs (default: rounds*local_epochs)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fusions", default=None,
                    help="comma-separated fusion methods to run (default: all methods in FUSIONS)")
    parser.add_argument("--no-fedavg", dest="fedavg", action="store_false",
                    help="skip the FedAvg baseline")
    parser.add_argument("--no-recalibrate", dest="recalibrate", action="store_false")
    parser.add_argument("--no-reference", dest="reference", action="store_false",
                    help="skip the full-model reference phase")
    parser.add_argument("--reset-results", action="store_true",
                    help="remove existing result CSV files before starting")
    parser.add_argument("--verbose", type=int, default=0,
                    help="model.fit verbosity (0 recommended)")
    args = parser.parse_args()

    if args.ref_epochs is None:
        args.ref_epochs = args.rounds * args.local_epochs

    check_no_layer_skips()

    if args.reset_results and RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.csv"):
            f.unlink()
        print(f"[reset] CSV files removed from {RESULTS_DIR}/")

    keras.utils.set_random_seed(RANDOM_SEED)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    print(f"TF {tf.__version__} | GPUs: {[g.name for g in gpus] or 'none (CPU)'}")
    print(f"Federated setup: {args.rounds} rounds x {args.local_epochs} epochs "
          f"| server fusion after every round | final evaluation only")
    print(f"Fusion module: {_FUSIONS_MODULE}")
    print(f"Results directory: {RESULTS_DIR}/")

    if args.fusions:
        requested_fusions = [f.strip() for f in args.fusions.split(",") if f.strip()]
        missing_fusions = [f for f in requested_fusions if f not in FUSIONS]
        if missing_fusions:
            raise ValueError(f"Unknown fusion methods: {missing_fusions}. "
                             f"Available: {list(FUSIONS)}")
        fusions = {k: FUSIONS[k] for k in requested_fusions}
    else:
        fusions = dict(FUSIONS)

    extra = {"rounds": args.rounds, "local_epochs": args.local_epochs}

    rng = np.random.default_rng(RANDOM_SEED)
    client_paths = build_client_paths(args.pattern, NUM_CLIENTS)
    train_sets, test_sets = [], []
    for i, path in enumerate(client_paths, 1):
        xc, yc = load_client(path, args.channels_first)
        (xtr, ytr), (xte, yte) = split_train_test(xc, yc, TEST_FRACTION, rng)
        train_sets.append((xtr, ytr)); test_sets.append((xte, yte))
        print(f"Client {i}: {len(ytr)} train / {len(yte)} test "
              f"({os.path.basename(path)})")

    reference_accuracies = []
    reference_accuracy = None
    if args.reference:
        print(f"\n{'='*70}\nPHASE 1 - REFERENCE: each client trains the full model "
              f"({args.ref_epochs} epochs, non-federated)\n{'='*70}")
        reference_frames = []
        for i, ((xtr, ytr), (xte, yte)) in enumerate(zip(train_sets, test_sets), 1):
            print(f"\n==> [REF] Client {i} | full model "
                  f"{GLOBAL_CONV_CHANNELS} + {GLOBAL_DENSE_UNITS} | n={len(ytr)}")
            m = build_server_model(args.lr, name=f"ref_client_{i}")
            m.fit(xtr, ytr, epochs=args.ref_epochs, batch_size=args.batch_size,
                  validation_data=(xte, yte), verbose=args.verbose)
            df_i, acc = evaluate_metrics(m, xte, yte, f"REF Client {i}")
            df_i.insert(0, "Test", f"Client {i}")
            df_i.insert(0, "Metodo", "REFERENCIA")
            reference_frames.append(df_i)
            reference_accuracies.append(acc)
            del m
            keras.backend.clear_session()

        append_summary_row("REFERENCIA", reference_frames,
                           extra={"rounds": 0, "local_epochs": args.ref_epochs})
        reference_accuracy = float(np.mean(reference_accuracies))
        print(f"\n>>> REFERENCE (mean across {NUM_CLIENTS} clients with full model): "
              f"{reference_accuracy*100:.2f}%")
    else:
        print(f"\n{'='*70}\nPHASE 1 - REFERENCE SKIPPED (--no-reference)\n{'='*70}")

    fusion_accuracies = {}
    fusion_mean_accuracy = {}
    submodel_accuracies = {}
    for fusion_name, fusion_fn in fusions.items():
        fusion_frames, fusion_accuracy_list, submodel_frames, submodel_accuracy_list = run_federated_fusion(
            fusion_name, fusion_fn, train_sets, test_sets, args)
        append_summary_row(fusion_name, fusion_frames, extra=extra)
        append_summary_row(f"SUBMODELOS_{fusion_name}", submodel_frames, extra=extra)
        fusion_accuracies[fusion_name] = fusion_accuracy_list
        fusion_mean_accuracy[fusion_name] = float(np.mean(fusion_accuracy_list))
        submodel_accuracies[fusion_name] = submodel_accuracy_list

    fedavg_accuracies = None
    if args.fedavg:
        fedavg_frames, fedavg_accuracy_list = run_fedavg(train_sets, test_sets, args)
        append_summary_row("FedAvg", fedavg_frames, extra=extra)
        fedavg_accuracies = fedavg_accuracy_list

    per_client_data = {
        "Cliente": [f"Client {i}" for i in range(1, NUM_CLIENTS + 1)],
        "N train": [len(y) for _, y in train_sets],
        "N test": [len(y) for _, y in test_sets],
    }
    if args.reference:
        per_client_data["Full reference (%)"] = [a * 100 for a in reference_accuracies]
    per_client_data.update({f"{k} (%)": [a * 100 for a in v]
                            for k, v in fusion_accuracies.items()})
    per_client_data.update({f"SUB {k} (%)": [a * 100 for a in v]
                            for k, v in submodel_accuracies.items()})
    if fedavg_accuracies is not None:
        per_client_data["FedAvg (%)"] = [a * 100 for a in fedavg_accuracies]
    per_client = pd.DataFrame(per_client_data)
    print("\n=== PER-CLIENT RESULTS ===")
    print(tabulate(per_client, headers="keys", tablefmt="github",
                   floatfmt=".2f", showindex=False))

    rows = []
    if args.reference:
        rows.append({"Metodo": "REFERENCIA (mean across clients, full model local)",
                     "Accuracy (%)": reference_accuracy * 100, "Delta vs REF (pp)": 0.0})
    for fusion_name in fusion_mean_accuracy:
        r = {"Metodo": f"Fusion {fusion_name}", "Accuracy (%)": fusion_mean_accuracy[fusion_name] * 100}
        if args.reference:
            r["Delta vs REF (pp)"] = (fusion_mean_accuracy[fusion_name] - reference_accuracy) * 100
        rows.append(r)
        rs = {"Metodo": f"Local submodels after rounds ({fusion_name})",
              "Accuracy (%)": float(np.mean(submodel_accuracies[fusion_name])) * 100}
        if args.reference:
            rs["Delta vs REF (pp)"] = (float(np.mean(submodel_accuracies[fusion_name])) - reference_accuracy) * 100
        rows.append(rs)
    if fedavg_accuracies is not None:
        r = {"Metodo": "FedAvg (full model)",
             "Accuracy (%)": float(np.mean(fedavg_accuracies)) * 100}
        if args.reference:
            r["Delta vs REF (pp)"] = (float(np.mean(fedavg_accuracies)) - reference_accuracy) * 100
        rows.append(r)
    comparison = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)

    print(f"\n=== FINAL COMPARISON ({args.rounds} rounds x "
          f"{args.local_epochs} epochs) ===")
    print(tabulate(comparison, headers="keys", tablefmt="github",
                   floatfmt=".2f", showindex=False))

    if fusion_mean_accuracy:
        best_fusion = max(fusion_mean_accuracy, key=fusion_mean_accuracy.get)
        reference_text = f" | reference: {reference_accuracy*100:.2f}%" if args.reference else ""
        fedavg_text = (f" | FedAvg: {np.mean(fedavg_accuracies)*100:.2f}%"
                  if fedavg_accuracies is not None else "")
        print(f"\nBest fusion: {best_fusion} ({fusion_mean_accuracy[best_fusion]*100:.2f}%){reference_text}{fedavg_text}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "COMPARATIVA.csv", index=False)
    print(f"\nGenerated files in '{RESULTS_DIR}/':")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
