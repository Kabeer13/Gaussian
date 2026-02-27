# Gaussian Naive Bayes App

This repository contains a simple implementation of Gaussian Naive Bayes and a small application
for training/evaluating it either via command line or using Streamlit.

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv env
.\env\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Running

### Command‑line interface

```powershell
python kbc.py
```

The script will prompt for dataset path, target and features, and test split. It then prints the
train/test percentages, confusion matrix, and classification report.

### Streamlit interface

```powershell
streamlit run kbc.py
```

Upload a CSV, choose the target/feature columns and test fraction, then click **Train & evaluate**.
Results appear directly in the browser.

## Notes

* The dataset must be a CSV with numeric feature columns.
* `requirements.txt` lists packages needed (`numpy`, `pandas`, `scikit-learn`, `streamlit`).
* Feel free to extend or convert this into a packaged module.
