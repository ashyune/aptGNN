# aptGNN
Detection of APTs using Stateful Heterogeneous GNNs on Provenance Graphs.

## Environment Setup

This project was developed and tested using Python 3.12 and PyTorch Geometric 2.x.

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

This repository does not include the DARPA TC datasets, generated feature files, trained models, or experimental outputs.

To reproduce the experiments:

1. Obtain the required DARPA TC datasets separately.
2. Place the datasets in the expected directory structure.
3. Run the preprocessing pipeline.
4. Train the model.
5. Evaluate the trained model.

---

## Execution Pipeline

### 1. Parse and preprocess the DARPA dataset

```bash
python scripts/data_process_train.py
python scripts/data_process_test.py
```

This generates the graph representation and feature-label files required for training and evaluation.

### 2. Train the model

```bash
python scripts/train_darpatc.py
```

This trains the Graph Neural Network on the processed training data.

### 3. Evaluate the model

```bash
python scripts/test_darpatc.py
```

This evaluates the trained model on the test dataset and generates anomaly predictions.

---

## Repository Contents

### Included

* Source code (`scripts/`)
* Ground truth files (`groundtruth/`)
* Environment specifications (`requirements.txt`, `environment.yml`)
* Documentation

### Excluded

* DARPA TC datasets
* Generated feature-label files
* Trained model checkpoints
* Experimental output files
* GraphChi-generated artifacts
