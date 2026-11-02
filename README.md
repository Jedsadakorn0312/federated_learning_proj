# Federated Learning for Molecular Property Prediction

This project investigates whether federated learning can predict molecular
bioactivity without centralizing the training data from participating clients.
Molecules are represented as graphs derived from SMILES strings, and the target
is the logarithmic potency value (pIC50).

## Research summary

- **Task:** Regression of molecular pIC50 values for drug-discovery datasets
  associated with diabetes, heart disease, and Alzheimer's disease.
- **Model:** A graph neural network using GATv2 attention layers, layer
  normalization, residual connections, and mean/max graph pooling.
- **Federated setup:** Flower coordinates multiple clients. FedAvg is used as
  the aggregation baseline, and the client training loop also supports the
  FedProx proximal term.
- **Experimental settings:** The scripts compare centralized training with
  federated training under separated disease distributions, IID partitions,
  and realistic non-IID quantity-skew partitions.
- **Evaluation:** Mean squared error (MSE), mean absolute error (MAE), and
  R2 are calculated on a shared validation/test split.
- **Hyperparameter search:** Optuna is used to explore hidden dimension,
  attention heads, number of layers, dropout, learning rate, and weight decay.

The experiments are intended to study privacy-preserving training behavior and
model performance. The results are not clinical evidence and must not be used
for diagnosis, treatment, or drug-safety decisions.

## Repository layout

- `FedProj/dataset.py` converts SMILES strings into PyTorch Geometric graphs.
- `FedProj/model.py` defines the GATv2 molecular graph model.
- `FedProj/client.py` implements local training, evaluation, and Flower client
  communication.
- `FedProj/main.py` runs the baseline federated simulation.
- `FedProj/experiment_2.py`, `experiment_3.py`, and `experiment_4.py` run
  centralized/federated comparison experiments.
- `FedProj/evaluation.py` calculates comparison metrics and creates plots.
- `FedProj/tune_model.py` runs Optuna hyperparameter tuning.

## Data and generated artifacts

The public repository intentionally excludes raw CSV datasets, processed
PyTorch datasets, trained model weights, logs, caches, and generated figures.
These files may contain research data or derived artifacts and should be
stored separately with appropriate access controls. To run the scripts,
provide the required local datasets under `FedProj/raw/` and use the file
names expected by the selected experiment.

## Reproducibility notes

The scripts use PyTorch, PyTorch Geometric, RDKit, Flower, scikit-learn,
Optuna, pandas, NumPy, matplotlib, seaborn, and Ray. Exact results depend on
the package versions, hardware, random seeds, data split, and Flower/Ray
configuration. Record those details alongside any future reported results.
