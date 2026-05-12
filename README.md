# Battery Contacting Quality Prediction

Public-safe portfolio version of a machine-learning project for industrial battery contacting quality analysis.

## Project overview

This project demonstrates a machine-learning pipeline for quality prediction from multi-channel industrial time-series data. The public version focuses on reusable engineering patterns rather than private project material: data loading, signal preprocessing, feature extraction, classical model evaluation, and LSTM-based sequence modeling.

## Project highlights

- Built a structured Python package for industrial sensor time-series analysis
- Implemented reusable preprocessing and feature extraction utilities
- Added Random Forest training/evaluation with stratified cross-validation
- Added an LSTM sequence classifier with oversampling and jitter augmentation
- Compared classical ML and sequence-modeling approaches under class imbalance
- Included runnable synthetic-data demos so reviewers can execute the workflow without private data
- Kept raw data, labels, reports, figures, partner details, and private results out of Git

## Tech stack

- Python
- NumPy, Pandas
- scikit-learn
- PyTorch
- SQLite

## Confidentiality note

The original project was completed in an academic/industry setting and may be subject to confidentiality constraints. For that reason, this repository intentionally excludes:

- raw measurement files
- label tables and experiment metadata
- project partner details
- original reports, slides, figures, and process images
- fold-level predictions, confusion matrices, and raw model artifacts

All runnable examples are sample-only implementations that use synthetic data. They are included to demonstrate the code structure under NDA/confidentiality constraints, not to reproduce or disclose private experiments.

The repository focuses on the technical pipeline: loading multi-channel time-series sensor data, preprocessing signals, extracting statistical features, and training/evaluating both classical ML and LSTM models for quality classification. Synthetic demos are included so the code can be reviewed and executed without private data.

## My contribution

- Data preprocessing pipeline for multi-channel time-series measurements
- Feature engineering for statistical signal descriptors
- Random Forest baseline training, validation, and evaluation
- LSTM sequence-model implementation with fold-safe augmentation
- Model comparison between classical ML and sequential deep learning approaches
- Result analysis and interpretation under class imbalance and confidentiality constraints
- Public-safe repository packaging using synthetic demos instead of private data

## Repository structure

```text
.
|-- examples/
|   |-- extract_features_from_private_data.py
|   |-- run_synthetic_demo.py
|   `-- run_synthetic_lstm_demo.py
|-- src/
|   `-- battery_quality/
|       |-- config.py
|       |-- data_loading.py
|       |-- features.py
|       |-- lstm.py
|       `-- modeling.py
|-- requirements.txt
`-- README.md
```

## Sample Code Notice

This repository contains a code-only, portfolio-safe sample version of the project. Due to NDA/confidentiality constraints, the original dataset, labels, reports, figures, partner-specific details, and actual experiment outputs are not included.

The runnable scripts use synthetic sample data only. They are intended to show the project structure, preprocessing approach, feature engineering, model training workflow, and LSTM implementation without disclosing private project material.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/run_synthetic_demo.py
python examples/run_synthetic_lstm_demo.py
```

Expected output: small cross-validation summaries for synthetic Random Forest and LSTM demos.

## What reviewers can evaluate

- Code organization for a small ML package
- Handling of private-data boundaries in a public portfolio repository
- Feature engineering design for time-series sensor channels
- Baseline model evaluation using scikit-learn
- LSTM implementation for sequence classification in PyTorch
- Comparative modeling workflow across Random Forest and LSTM approaches
- Reproducible synthetic demos that do not require access to private files

## Private data workflow

Raw data is not part of this repository. If you have authorized access to private data, keep it outside Git and point the extraction script to it with environment variables:

```bash
set PRIVATE_DATA_ROOT=D:\path\to\private\data
set PRIVATE_LABEL_TABLE=D:\path\to\private\labels.xlsx
python examples/extract_features_from_private_data.py
```

The resulting feature table should also remain private unless it has been explicitly cleared for publication.

## Notes for reviewers

This repo is designed to show the engineering approach without exposing confidential project material. The public code demonstrates the reusable parts of the work: data ingestion patterns, signal preprocessing, feature extraction, and model evaluation.

## License

No open-source license is provided. This repository is shared for portfolio review only, and the original project context remains subject to confidentiality constraints.
