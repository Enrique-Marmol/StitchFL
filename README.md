# Federated Submodel Training Benchmarks

This repository contains the code used to evaluate heterogeneous federated submodel training across different neural network architectures and datasets.

The experiments include:

* **MLP** experiments on the **UCI Letter Recognition** dataset.
* **CNN** experiments on **CIFAR-10**.
* **LSTM** experiments on the **PLEIADATA** time-series dataset.
* **Membership Inference Attack (MIA)** experiments for evaluating privacy leakage under different model capacities.

The federated experiments compare heterogeneous submodel approaches with standard **FedAvg** and other submodel-training baselines.

## Repository structure

```text
.
├── Datasets/
│   ├── letter+recognition/
│   ├── cifar/
│   └── pleiadata/
│
├── federated_mlp_benchmark.py
├── federated_cnn_benchmark.py
├── federated_lstm_benchmark.py
├── fusions.py
├── fusions_fl.py
└── mia_suite_classification.py
```

The `Datasets/` directory contains the **three datasets required to reproduce the experiments**, so no additional dataset download is required if this folder is already included.

## Requirements

The code is implemented in Python and mainly requires TensorFlow, NumPy, pandas, scikit-learn and SciPy.

Install the required packages with:

```bash
pip install tensorflow numpy pandas scikit-learn scipy tabulate matplotlib
```

A GPU can be used when available, although the experiments can also be executed on CPU.

## MLP experiments

The MLP benchmark uses the **UCI Letter Recognition** dataset and compares:

* StitchFL
* HaloFL
* FedGSE
* SubFLOT
* FedAvg

By default, only **StitchFL** is executed:

```bash
python federated_mlp_benchmark.py
```

To execute all methods:

```bash
python federated_mlp_benchmark.py --methods all
```

Specific methods can also be selected:

```bash
python federated_mlp_benchmark.py --methods stitchfl fedavg
```

Multiple independent repetitions can be performed with:

```bash
python federated_mlp_benchmark.py --methods all --reps 5
```

Some StitchFL parameters can be configured directly from the command line:

```bash
python federated_mlp_benchmark.py \
    --methods stitchfl \
    --stitch-mode factor \
    --distill-scope all \
    --anchors 20000
```

Recalibration or ensemble evaluation can be disabled with:

```bash
--no-recalibration
--no-ensemble
```

Results are stored under:

```text
resultados/FL/MLP/letter_recognition/
```

## CNN experiments

The CNN benchmark uses **CIFAR-10** partitioned among the federated clients.

Run the complete benchmark with:

```bash
python federated_cnn_benchmark.py
```

The number of federated rounds and local epochs can be modified with:

```bash
python federated_cnn_benchmark.py \
    --rounds 30 \
    --local-epochs 5
```

A specific set of fusion methods can be executed with:

```bash
python federated_cnn_benchmark.py \
    --fusions StitchFL,HaloFL,FedGSE,SubFLOT
```

FedAvg and the full-model reference experiment can be disabled when they are not required:

```bash
python federated_cnn_benchmark.py --no-fedavg --no-reference
```

Existing result CSV files can be cleared before a new experiment with:

```bash
python federated_cnn_benchmark.py --reset-results
```

The default CIFAR-10 partitions are read from:

```text
Datasets/cifar/cifar10_part_*.csv
```

and the results are stored in:

```text
results/CNN/
```

## LSTM experiments

The LSTM benchmark evaluates heterogeneous federated learning on the **PLEIADATA** time-series dataset.

Run it with:

```bash
python federated_lstm_benchmark.py
```

The script automatically executes:

1. Independent full-model training for each client.
2. Heterogeneous federated submodel training.
3. FedAvg with the complete model.
4. Evaluation on each client's test data.
5. Storage of the final metrics.

The data are loaded from:

```text
Datasets/pleiadata/
├── data-model-consumoA-60T.csv
├── data-model-consumoB-60T.csv
└── data-model-consumoC-60T.csv
```

The experiment configuration, including the number of rounds, local epochs, model size and client capacity, can be modified at the beginning of `federated_lstm_benchmark.py`.

Results are stored under:

```text
results/LSTM/
```

## Membership Inference Attacks

The repository also includes a Membership Inference Attack benchmark:

```bash
python mia_suite_classification.py
```

The implemented attacks are:

* Yeom
* LiRA
* RMIA
* Attack-R
* Quantile
* TrajectoryMIA (TMIA)

The default experiment evaluates several MLP architectures with different capacities.

For a quick test of the pipeline, use:

```bash
python mia_suite_classification.py --quick
```

A custom set of architectures can be evaluated with:

```bash
python mia_suite_classification.py \
    --archs 16 64,32 256,128,64
```

The number of shadow models can be changed with:

```bash
python mia_suite_classification.py --n_shadows 64
```

TMIA is computationally more expensive and can be disabled with:

```bash
python mia_suite_classification.py --skip_tmia
```

GPU execution can be enabled with:

```bash
python mia_suite_classification.py --gpu
```

By default, the MIA results are stored in:

```text
results/mia_classification/
```

## Dataset organization

The repository expects the following dataset structure:

```text
Datasets/
├── letter+recognition/
│   └── letter-recognition.data
│
├── cifar_partes/
│   ├── cifar10_part_1.csv
│   ├── cifar10_part_2.csv
│   ├── ...
│   └── cifar10_part_5.csv
│
└── pleiadata/
    ├── data-model-consumoA-60T.csv
    ├── data-model-consumoB-60T.csv
    └── data-model-consumoC-60T.csv
```

All paths are defined relative to the directory from which the scripts are executed. Therefore, the scripts should normally be launched from the **root directory of the repository**.

For example:

```bash
cd path/to/repository
python federated_mlp_benchmark.py
```

## Main configuration

The main experimental parameters, such as:

* global model architecture,
* client submodel sizes,
* neuron/channel ranges,
* number of federated rounds,
* number of local epochs,
* batch size,
* learning rate,
* number of anchors,

are defined near the beginning of each benchmark script and can be modified to reproduce different experimental configurations.

## Output

The scripts print the evolution of the federated training process and the final evaluation metrics to the terminal.

Depending on the experiment, the reported metrics include classification metrics such as:

```text
Accuracy
Precision
Recall
F1-score
MCC
```

or regression metrics such as:

```text
RMSE
MAE
R²
NMBE
CV(RMSE)
```

The aggregated results are also stored as CSV files in the corresponding results directory.

