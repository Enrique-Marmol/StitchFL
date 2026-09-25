"""Fusion strategies for round-based heterogeneous CNN federated learning."""

import os

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def conv_layers_of(model):
    return [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]


def bn_layers_of(model):
    return [layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)]


def dense_layers_of(model):
    return [layer for layer in model.layers if isinstance(layer, layers.Dense)]


def safe_div(numerator, denominator):
    return numerator / np.where(denominator == 0, 1, denominator)


def _snapshot_server(server):
    return {
        "conv": [layer.get_weights()[0].copy() for layer in conv_layers_of(server)],
        "bn": [[w.copy() for w in layer.get_weights()] for layer in bn_layers_of(server)],
        "dense": [[w.copy() for w in layer.get_weights()] for layer in dense_layers_of(server)],
    }


def _client_index_maps(conv_ranges, dense_ranges, n_classes):
    conv_indices = []
    for layer_idx, (start, end) in enumerate(conv_ranges):
        input_idx = np.arange(3) if layer_idx == 0 else np.arange(*conv_ranges[layer_idx - 1])
        conv_indices.append((input_idx, np.arange(start, end)))

    dense_indices = []
    input_idx = np.arange(*conv_ranges[-1])
    for start, end in dense_ranges:
        if end <= start:
            continue
        output_idx = np.arange(start, end)
        dense_indices.append((input_idx, output_idx))
        input_idx = output_idx
    dense_indices.append((input_idx, np.arange(n_classes)))
    return conv_indices, dense_indices


def _init_accumulators(server):
    conv_layers = conv_layers_of(server)
    bn_layers = bn_layers_of(server)
    dense_layers = dense_layers_of(server)

    conv_sum = [np.zeros_like(layer.get_weights()[0]) for layer in conv_layers]
    conv_count = [np.zeros_like(x) for x in conv_sum]
    bn_sum = [[np.zeros_like(w) for w in layer.get_weights()] for layer in bn_layers]
    bn_count = [[np.zeros_like(w) for w in layer.get_weights()] for layer in bn_layers]

    dense_w_sum, dense_w_count, dense_b_sum, dense_b_count = [], [], [], []
    for layer in dense_layers:
        weights, bias = layer.get_weights()
        dense_w_sum.append(np.zeros_like(weights))
        dense_w_count.append(np.zeros_like(weights))
        dense_b_sum.append(np.zeros_like(bias))
        dense_b_count.append(np.zeros_like(bias))

    return (
        conv_layers, bn_layers, dense_layers,
        conv_sum, conv_count, bn_sum, bn_count,
        dense_w_sum, dense_w_count, dense_b_sum, dense_b_count,
    )


def _set_averaged_weights(server, accumulators, previous):
    (
        conv_layers, bn_layers, dense_layers,
        conv_sum, conv_count, bn_sum, bn_count,
        dense_w_sum, dense_w_count, dense_b_sum, dense_b_count,
    ) = accumulators

    for idx, layer in enumerate(conv_layers):
        layer.set_weights([
            np.where(
                conv_count[idx] > 0,
                safe_div(conv_sum[idx], conv_count[idx]),
                previous["conv"][idx],
            )
        ])

    for idx, layer in enumerate(bn_layers):
        layer.set_weights([
            np.where(
                bn_count[idx][weight_idx] > 0,
                safe_div(bn_sum[idx][weight_idx], bn_count[idx][weight_idx]),
                previous["bn"][idx][weight_idx],
            )
            for weight_idx in range(len(bn_sum[idx]))
        ])

    for idx, layer in enumerate(dense_layers):
        layer.set_weights([
            np.where(
                dense_w_count[idx] > 0,
                safe_div(dense_w_sum[idx], dense_w_count[idx]),
                previous["dense"][idx][0],
            ),
            np.where(
                dense_b_count[idx] > 0,
                safe_div(dense_b_sum[idx], dense_b_count[idx]),
                previous["dense"][idx][1],
            ),
        ])


def _recalibrate_bn(server, client_x, sample_counts=None, n_target=8000, batch=256, verbose=True):
    if client_x is None:
        return
    if sample_counts is None:
        sample_counts = [len(x) for x in client_x]

    total_samples = float(sum(sample_counts))
    pooled = []
    for x_data, count in zip(client_x, sample_counts):
        size = max(1, int(n_target * count / total_samples))
        indices = np.random.RandomState(0).choice(len(x_data), min(size, len(x_data)), replace=False)
        pooled.append(x_data[indices])

    x_pool = np.concatenate(pooled)
    np.random.RandomState(1).shuffle(x_pool)
    bn_layers = bn_layers_of(server)
    if not bn_layers:
        return

    extractor = Model(server.inputs, [layer.input for layer in bn_layers])
    running_mean = [np.zeros_like(layer.get_weights()[2]) for layer in bn_layers]
    running_var = [np.zeros_like(layer.get_weights()[3]) for layer in bn_layers]
    seen = 0
    num_batches = (len(x_pool) + batch - 1) // batch

    for batch_idx in range(num_batches):
        x_batch = x_pool[batch_idx * batch:(batch_idx + 1) * batch]
        if not len(x_batch):
            break
        outputs = extractor(x_batch, training=False)
        if len(bn_layers) == 1:
            outputs = [outputs]
        batch_size = len(x_batch)
        for idx, activation in enumerate(outputs):
            activation = np.asarray(activation)
            mean = activation.mean(axis=(0, 1, 2))
            var = activation.var(axis=(0, 1, 2))
            running_mean[idx] = (running_mean[idx] * seen + mean * batch_size) / (seen + batch_size)
            running_var[idx] = (running_var[idx] * seen + var * batch_size) / (seen + batch_size)
        seen += batch_size

    for idx, layer in enumerate(bn_layers):
        gamma, beta, _, _ = layer.get_weights()
        layer.set_weights([gamma, beta, running_mean[idx], running_var[idx]])

    if verbose:
        print(f"    BatchNorm recalibrated with {seen} samples ({num_batches} batches)")


def compute_uil_conv(model, x, conv_ranges, alpha=0.5, theta=0.0, max_samples=1000):
    """Compute channel importance from activation magnitude and frequency."""
    activation_layers = [layer for layer in model.layers if isinstance(layer, layers.Activation)]
    conv_activations = activation_layers[:len(conv_ranges)]
    if not conv_activations:
        return [np.ones(end - start) for start, end in conv_ranges]

    extractor = Model(model.inputs, [layer.output for layer in conv_activations])
    outputs = extractor.predict(x[:max_samples], verbose=0)
    if len(conv_activations) == 1:
        outputs = [outputs]

    scores = []
    for activation in outputs:
        activation = np.asarray(activation)
        mask = activation > theta
        ratio = mask.mean(axis=(0, 1, 2))
        magnitude = (activation * mask).sum(axis=(0, 1, 2))
        ratio /= ratio.max() if ratio.max() > 0 else 1.0
        magnitude /= magnitude.max() if magnitude.max() > 0 else 1.0
        scores.append(alpha * magnitude + (1.0 - alpha) * ratio)
    return scores


def fuse_stitchfl(
    clients, server, conv_ranges, dense_ranges, n_classes=10,
    sample_counts=None, client_x=None, recalibrate_bn=True,
    fill_mode="template", lam=0.5, imp_alpha=0.5, verbose=True, **kwargs,
):
    """Sample-weighted positional fusion with soft APoZ channel weighting."""
    del kwargs
    lam = np.clip(float(os.environ.get("STITCHFL_LAM", lam)), 0.0, 1.0)
    sample_counts = sample_counts or [1.0] * len(clients)

    server_conv = conv_layers_of(server)
    server_bn = bn_layers_of(server)
    server_dense = dense_layers_of(server)
    previous = _snapshot_server(server)

    conv_sum = [np.zeros_like(layer.get_weights()[0]) for layer in server_conv]
    conv_count = [np.zeros_like(x) for x in conv_sum]
    bn_sum = [[np.zeros_like(w) for w in layer.get_weights()[:2]] for layer in server_bn]
    bn_count = [[np.zeros_like(w) for w in layer.get_weights()[:2]] for layer in server_bn]
    dense_w_sum, dense_w_count, dense_b_sum, dense_b_count = [], [], [], []
    for layer in server_dense:
        weights, bias = layer.get_weights()
        dense_w_sum.append(np.zeros_like(weights)); dense_w_count.append(np.zeros_like(weights))
        dense_b_sum.append(np.zeros_like(bias)); dense_b_count.append(np.zeros_like(bias))

    for client_idx, (client, client_conv, client_dense, count) in enumerate(
        zip(clients, conv_ranges, dense_ranges, sample_counts)
    ):
        conv_idx, dense_idx = _client_index_maps(client_conv, client_dense, n_classes)
        client_conv_layers = conv_layers_of(client)
        client_bn_layers = bn_layers_of(client)
        client_dense_layers = dense_layers_of(client)

        if lam > 0 and client_x is not None:
            raw_scores = compute_uil_conv(client, client_x[client_idx], client_conv, alpha=imp_alpha)
            importance = []
            for score in raw_scores:
                score = np.asarray(score, dtype=np.float64)
                score /= score.max() if score.max() > 0 else 1.0
                importance.append((1.0 - lam) + lam * score)
        else:
            importance = [np.ones(end - start) for start, end in client_conv]

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(conv_idx, client_conv_layers)):
            weights = layer.get_weights()[0]
            channel_weight = float(count) * importance[idx]
            target = np.ix_(range(weights.shape[0]), range(weights.shape[1]), input_idx, output_idx)
            conv_sum[idx][target] += weights * channel_weight[None, None, None, :]
            conv_count[idx][target] += np.broadcast_to(channel_weight[None, None, None, :], weights.shape)

        for idx, ((start, end), layer) in enumerate(zip(client_conv, client_bn_layers)):
            indices = np.arange(start, end)
            channel_weight = float(count) * importance[idx]
            gamma, beta = layer.get_weights()[:2]
            bn_sum[idx][0][indices] += gamma * channel_weight
            bn_count[idx][0][indices] += channel_weight
            bn_sum[idx][1][indices] += beta * channel_weight
            bn_count[idx][1][indices] += channel_weight

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(dense_idx, client_dense_layers)):
            weights, bias = layer.get_weights()
            target = np.ix_(input_idx, output_idx)
            dense_w_sum[idx][target] += weights * float(count)
            dense_w_count[idx][target] += float(count)
            dense_b_sum[idx][output_idx] += bias * float(count)
            dense_b_count[idx][output_idx] += float(count)

    for idx, layer in enumerate(server_conv):
        averaged = safe_div(conv_sum[idx], conv_count[idx])
        fallback = previous["conv"][idx] if fill_mode == "template" else np.zeros_like(averaged)
        covered = conv_count[idx] > 0
        layer.set_weights([np.where(covered, averaged, fallback)])
        if verbose:
            print(f"    conv{idx}: {covered.mean() * 100:.0f}% coverage (uncovered -> {fill_mode})")

    for idx, layer in enumerate(server_bn):
        gamma = np.where(bn_count[idx][0] > 0, safe_div(bn_sum[idx][0], bn_count[idx][0]), previous["bn"][idx][0])
        beta = np.where(bn_count[idx][1] > 0, safe_div(bn_sum[idx][1], bn_count[idx][1]), previous["bn"][idx][1])
        _, _, moving_mean, moving_var = layer.get_weights()
        layer.set_weights([gamma, beta, moving_mean, moving_var])

    for idx, layer in enumerate(server_dense):
        weights = safe_div(dense_w_sum[idx], dense_w_count[idx])
        bias = safe_div(dense_b_sum[idx], dense_b_count[idx])
        fallback_w = previous["dense"][idx][0] if fill_mode == "template" else np.zeros_like(weights)
        fallback_b = previous["dense"][idx][1] if fill_mode == "template" else np.zeros_like(bias)
        layer.set_weights([
            np.where(dense_w_count[idx] > 0, weights, fallback_w),
            np.where(dense_b_count[idx] > 0, bias, fallback_b),
        ])

    if recalibrate_bn and client_x is not None:
        _recalibrate_bn(server, client_x, sample_counts, verbose=verbose)
        if verbose:
            print(f"    soft channel weighting (lambda={lam}) + BatchNorm recalibration")
    return server


def fuse_fedgse(
    clients, server, conv_ranges, dense_ranges, n_classes=10,
    sample_counts=None, verbose=True, **kwargs,
):
    """Sample-weighted partial delta aggregation."""
    del kwargs
    sample_counts = sample_counts or [1.0] * len(clients)
    total = float(sum(sample_counts))

    server_conv = conv_layers_of(server)
    server_bn = bn_layers_of(server)
    server_dense = dense_layers_of(server)
    old_conv = [layer.get_weights()[0].copy() for layer in server_conv]
    new_conv = [x.copy() for x in old_conv]
    old_bn = [[w.copy() for w in layer.get_weights()] for layer in server_bn]
    new_bn = [[w.copy() for w in values] for values in old_bn]
    old_dense = [[w.copy() for w in layer.get_weights()] for layer in server_dense]
    new_dense = [[w.copy() for w in values] for values in old_dense]

    for client, client_conv, client_dense, count in zip(clients, conv_ranges, dense_ranges, sample_counts):
        coefficient = count / total
        conv_idx, dense_idx = _client_index_maps(client_conv, client_dense, n_classes)

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(conv_idx, conv_layers_of(client))):
            weights = layer.get_weights()[0]
            target = np.ix_(range(weights.shape[0]), range(weights.shape[1]), input_idx, output_idx)
            new_conv[idx][target] += coefficient * (weights - old_conv[idx][target])

        for idx, ((start, end), layer) in enumerate(zip(client_conv, bn_layers_of(client))):
            indices = np.arange(start, end)
            for weight_idx, weight in enumerate(layer.get_weights()):
                new_bn[idx][weight_idx][indices] += coefficient * (weight - old_bn[idx][weight_idx][indices])

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(dense_idx, dense_layers_of(client))):
            weights, bias = layer.get_weights()
            target = np.ix_(input_idx, output_idx)
            new_dense[idx][0][target] += coefficient * (weights - old_dense[idx][0][target])
            new_dense[idx][1][output_idx] += coefficient * (bias - old_dense[idx][1][output_idx])

    for idx, layer in enumerate(server_conv):
        layer.set_weights([new_conv[idx]])
    for idx, layer in enumerate(server_bn):
        layer.set_weights(new_bn[idx])
    for idx, layer in enumerate(server_dense):
        layer.set_weights(new_dense[idx])

    if verbose:
        weights = [count / total for count in sample_counts]
        print(f"    sample-weighted delta aggregation {[f'{w:.2f}' for w in weights]}")
    return server


def fuse_halofl(
    clients, server, conv_ranges, dense_ranges, n_classes=10,
    client_x=None, alpha=0.5, verbose=True, **kwargs,
):
    """UIL-weighted positional aggregation."""
    del kwargs
    previous = _snapshot_server(server)
    accumulators = _init_accumulators(server)
    (
        _, _, _, conv_sum, conv_count, bn_sum, bn_count,
        dense_w_sum, dense_w_count, dense_b_sum, dense_b_count,
    ) = accumulators

    for client_idx, (client, client_conv, client_dense) in enumerate(zip(clients, conv_ranges, dense_ranges)):
        scores = (
            compute_uil_conv(client, client_x[client_idx], client_conv, alpha=alpha)
            if client_x is not None
            else [np.ones(end - start) for start, end in client_conv]
        )
        conv_idx, dense_idx = _client_index_maps(client_conv, client_dense, n_classes)

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(conv_idx, conv_layers_of(client))):
            weights = layer.get_weights()[0]
            channel_weight = np.maximum(scores[idx], 1e-3)
            target = np.ix_(range(weights.shape[0]), range(weights.shape[1]), input_idx, output_idx)
            conv_sum[idx][target] += weights * channel_weight[None, None, None, :]
            conv_count[idx][target] += np.broadcast_to(channel_weight[None, None, None, :], weights.shape)

        for idx, ((start, end), layer) in enumerate(zip(client_conv, bn_layers_of(client))):
            indices = np.arange(start, end)
            channel_weight = np.maximum(scores[idx], 1e-3)
            for weight_idx, weight in enumerate(layer.get_weights()):
                bn_sum[idx][weight_idx][indices] += weight * channel_weight
                bn_count[idx][weight_idx][indices] += channel_weight

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(dense_idx, dense_layers_of(client))):
            weights, bias = layer.get_weights()
            target = np.ix_(input_idx, output_idx)
            dense_w_sum[idx][target] += weights
            dense_w_count[idx][target] += 1
            dense_b_sum[idx][output_idx] += bias
            dense_b_count[idx][output_idx] += 1

    _set_averaged_weights(server, accumulators, previous)
    if verbose:
        print("    UIL-weighted aggregation; uncovered parameters keep previous global values")
    return server


def sinkhorn(mu, nu, cost, reg=0.05, n_iter=300, tol=1e-9):
    cost = np.asarray(cost, dtype=np.float64)
    kernel = np.exp(-(cost / (cost.max() + 1e-12)) / reg) + 1e-300
    u, v = np.ones_like(mu), np.ones_like(nu)
    for _ in range(n_iter):
        previous_u = u
        u = mu / (kernel @ v + 1e-300)
        v = nu / (kernel.T @ u + 1e-300)
        if np.max(np.abs(u - previous_u)) < tol:
            break
    return u[:, None] * kernel * v[None, :]


def normalize_otmap_sum(transport):
    column_sum = transport.sum(axis=0, keepdims=True)
    column_sum[column_sum == 0] = 1.0
    return transport / column_sum


def cost_matrix(source, target):
    source_norm = (source ** 2).sum(axis=1, keepdims=True)
    target_norm = (target ** 2).sum(axis=1, keepdims=True).T
    distance = np.maximum(source_norm + target_norm - 2.0 * source @ target.T, 0.0)
    return np.sqrt(distance + 1e-12)


def fuse_subflot(
    clients, server, conv_ranges, dense_ranges, n_classes=10,
    reg=0.05, verbose=True, **kwargs,
):
    """Align client channels with Sinkhorn OT before positional averaging."""
    del kwargs
    previous = _snapshot_server(server)
    accumulators = _init_accumulators(server)
    (
        _, _, _, conv_sum, conv_count, bn_sum, bn_count,
        dense_w_sum, dense_w_count, dense_b_sum, dense_b_count,
    ) = accumulators

    for client, client_conv, client_dense in zip(clients, conv_ranges, dense_ranges):
        conv_idx, dense_idx = _client_index_maps(client_conv, client_dense, n_classes)

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(conv_idx, conv_layers_of(client))):
            weights = layer.get_weights()[0]
            target = np.ix_(range(weights.shape[0]), range(weights.shape[1]), input_idx, output_idx)
            server_block = conv_sum[idx][target]
            source = weights.reshape(-1, weights.shape[3]).T
            target_channels = server_block.reshape(-1, server_block.shape[3]).T

            if np.abs(target_channels).sum() > 0:
                mu = np.ones(source.shape[0]) / source.shape[0]
                nu = np.ones(target_channels.shape[0]) / target_channels.shape[0]
                transport = normalize_otmap_sum(sinkhorn(mu, nu, cost_matrix(source, target_channels), reg=reg))
                weights = np.tensordot(weights, transport, axes=([3], [0]))

            conv_sum[idx][target] += weights
            conv_count[idx][target] += 1

        for idx, ((start, end), layer) in enumerate(zip(client_conv, bn_layers_of(client))):
            indices = np.arange(start, end)
            for weight_idx, weight in enumerate(layer.get_weights()):
                bn_sum[idx][weight_idx][indices] += weight
                bn_count[idx][weight_idx][indices] += 1

        for idx, ((input_idx, output_idx), layer) in enumerate(zip(dense_idx, dense_layers_of(client))):
            weights, bias = layer.get_weights()
            target = np.ix_(input_idx, output_idx)
            dense_w_sum[idx][target] += weights
            dense_w_count[idx][target] += 1
            dense_b_sum[idx][output_idx] += bias
            dense_b_count[idx][output_idx] += 1

    _set_averaged_weights(server, accumulators, previous)
    if verbose:
        print(f"    OT channel alignment (Sinkhorn, reg={reg}); uncovered parameters keep previous global values")
    return server


FUSIONS = {
    "StitchFL": fuse_stitchfl,
    "FedGSE": fuse_fedgse,
    "HaloFL": fuse_halofl,
    "SubFLOT": fuse_subflot,
}
