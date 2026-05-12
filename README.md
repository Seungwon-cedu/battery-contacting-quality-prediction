# Battery Contacting Quality Prediction

Public-safe portfolio version of a machine-learning project for industrial battery contacting quality analysis.

## Confidentiality note

The original project was completed in an academic/industry setting and may be subject to confidentiality constraints. For that reason, this repository intentionally excludes:

- raw measurement files
- label tables and experiment metadata
- project partner details
- original reports, slides, figures, and process images
- fold-level predictions, confusion matrices, and raw model artifacts

The repository focuses on the technical pipeline: loading multi-channel time-series sensor data, preprocessing signals, extracting statistical features, and training/evaluating both classical ML and LSTM models for quality classification. Synthetic demos are included so the code can be reviewed and executed without private data.

## Technical scope

- SQLite-based measurement ingestion
- multi-channel signal trimming and smoothing
- feature engineering for time-series sensor channels
- class-imbalance-aware model training
- stratified cross-validation
- Random Forest baseline and hyperparameter tuning
- LSTM sequence model with oversampling and jitter augmentation
- reusable project structure for private data kept outside Git

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

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/run_synthetic_demo.py
python examples/run_synthetic_lstm_demo.py
```

Expected output: small cross-validation summaries for synthetic Random Forest and LSTM demos.

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
